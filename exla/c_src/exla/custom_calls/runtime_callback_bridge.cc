#include "runtime_callback_bridge.h"

#include <cstring>

namespace exla {

namespace callback_bridge {

fine::Ok<> runtime_callback_reply(ErlNifEnv *env,
                                  fine::ResourcePtr<Pending> pending,
                                  fine::Atom status, fine::Term result) {
  deliver_reply(env, pending, status, result);
  return fine::Ok();
}

void deliver_reply(ErlNifEnv *env, fine::ResourcePtr<Pending> pending,
                   fine::Atom status, fine::Term result_term) {
  Result cb_result;

  if (status == "ok") {
    // Successful reply: result_term is a list of binaries that we decode into
    // raw byte vectors via Fine and copy directly into the registered output
    // buffers.
    try {
      auto payloads = fine::decode<std::vector<ErlNifBinary>>(env, result_term);

      std::lock_guard<std::mutex> lock(pending->mu);

      if (payloads.size() != pending->outputs.size()) {
        cb_result.ok = false;
        cb_result.error =
            "mismatched number of callback outputs vs registered buffers";
      } else {
        cb_result.ok = true;

        for (size_t i = 0; i < payloads.size(); ++i) {
          const ErlNifBinary &bytes = payloads[i];
          auto &out_buf = pending->outputs[i];

          if (bytes.size != out_buf.size) {
            cb_result.ok = false;
            cb_result.error =
                "callback returned binary of unexpected size for result buffer";
            break;
          }

          if (out_buf.size > 0) {
            std::memcpy(out_buf.data, bytes.data, out_buf.size);
          }
        }
      }
    } catch (const std::exception &e) {
      cb_result.ok = false;
      cb_result.error =
          std::string("failed to decode Elixir callback outputs: ") + e.what();
    }
  } else {
    // Error reply: result_term is expected to be {kind_atom, message :: binary}
    cb_result.ok = false;

    try {
      auto decoded =
          fine::decode<std::tuple<fine::Atom, ErlNifBinary>>(env, result_term);
      fine::Atom kind = std::get<0>(decoded);
      ErlNifBinary msg_bin = std::get<1>(decoded);

      cb_result.error =
          "elixir callback returned " + kind.to_string() + ": " +
          std::string(reinterpret_cast<const char *>(msg_bin.data),
                      msg_bin.size);
    } catch (const std::exception &) {
      cb_result.error = "elixir callback returned error";
    }
  }

  {
    std::lock_guard<std::mutex> lock(pending->mu);
    pending->result = std::move(cb_result);
    pending->done = true;
  }

  pending->cv.notify_one();
}

Result InvokeRuntimeCallback(
    xla::ffi::Span<const int64_t> callback_id_words, uint64_t callback_id_size,
    const std::vector<Arg> &inputs,
    const std::vector<OutputBuffer> &outputs,
    const uint8_t *pid_data, size_t pid_size) {
  auto pending = fine::make_resource<Pending>(outputs);

  ErlNifEnv *msg_env = enif_alloc_env();

  // Decode the callback server PID from the serialized binary.
  // The PID was serialized on the Elixir side via :erlang.term_to_binary/1.
  ERL_NIF_TERM pid_term;
  if (!enif_binary_to_term(msg_env, pid_data, pid_size, &pid_term, 0)) {
    enif_free_env(msg_env);
    Result res;
    res.ok = false;
    res.error = "failed to decode callback server PID from input tensor";
    return res;
  }

  ErlNifPid target_pid;
  if (!enif_get_local_pid(msg_env, pid_term, &target_pid)) {
    enif_free_env(msg_env);
    Result res;
    res.ok = false;
    res.error = "callback server PID is not a valid local PID";
    return res;
  }

  // Reinterpret the 64-bit words as a contiguous byte buffer and use the
  // original (unpadded) size when decoding the callback id term.
  if (callback_id_size > callback_id_words.size() * sizeof(int64_t)) {
    enif_free_env(msg_env);
    Result res;
    res.ok = false;
    res.error = "inconsistent callback id size";
    return res;
  }

  const unsigned char *id_bytes =
      reinterpret_cast<const unsigned char *>(callback_id_words.begin());

  ERL_NIF_TERM callback_id_term;
  if (!enif_binary_to_term(msg_env, id_bytes, callback_id_size,
                           &callback_id_term, 0)) {
    enif_free_env(msg_env);
    Result res;
    res.ok = false;
    res.error = "failed to decode callback id term";
    return res;
  }

  // Encode arguments as [{bin, %EXLA.Typespec{}}, ...]. We currently send
  // plain binaries because the BEAM callback needs to own the data lifetime.
  std::vector<std::tuple<fine::Term,
                         std::tuple<xla::ffi::DataType, std::vector<int64_t>>>>
      args_terms;
  args_terms.reserve(inputs.size());

  for (const auto &tensor : inputs) {
    fine::Term bin_term = fine::make_new_binary(
        msg_env, reinterpret_cast<const char *>(tensor.data),
        tensor.size_bytes);

    // Build an %EXLA.Typespec{} directly from the ffi::DataType and dims via
    // Fine's encoder defined in exla_nif_util.h.
    auto arg_tuple =
        std::make_tuple(bin_term, std::make_tuple(tensor.dtype, tensor.dims));

    args_terms.push_back(arg_tuple);
  }

  auto msg = std::make_tuple(fine::Atom("exla_runtime_call"),
                             fine::Term(callback_id_term), args_terms, pending);

  // Send directly to the callback server PID extracted from the input tensor.
  enif_send(msg_env, &target_pid, msg_env, fine::encode(msg_env, msg));
  enif_free_env(msg_env);

  std::unique_lock<std::mutex> lock(pending->mu);
  pending->cv.wait(lock, [&pending] { return pending->done; });

  return pending->result;
}

} // namespace callback_bridge

} // namespace exla
