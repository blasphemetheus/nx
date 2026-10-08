defmodule Nx.Defn.CheckpointFuzzTest.Program do
  @moduledoc false
  # A program is a list of steps. A step is an op index or a nested list of
  # steps wrapped in a checkpoint. Every op is smooth on all reals and does
  # not grow fast, so chains of up to ten stay finite in f64.
  #
  # This lives outside the test module so defn functions in the test can
  # call it: defn only allows calls into other modules.

  import Nx.Defn.Kernel, only: [checkpoint: 2]

  @op_count 12
  def op_count, do: @op_count

  def run(steps, x), do: Enum.reduce(steps, x, &step/2)

  def step(i, x) when is_integer(i), do: op(i, x)
  def step({:ckpt, steps}, x), do: checkpoint(x, fn x -> run(steps, x) end)

  def strip(steps) do
    Enum.flat_map(steps, fn
      {:ckpt, inner} -> strip(inner)
      i -> [i]
    end)
  end

  def apply_steps(x, steps, true), do: checkpoint(x, fn x -> run(steps, x) end)
  def apply_steps(x, steps, false), do: run(steps, x)

  defp op(0, x), do: Nx.sin(x)
  defp op(1, x), do: Nx.cos(x)
  defp op(2, x), do: Nx.tanh(x)
  defp op(3, x), do: Nx.atan(x)
  defp op(4, x), do: Nx.sigmoid(x)
  defp op(5, x), do: Nx.multiply(x, x)
  defp op(6, x), do: Nx.add(x, 1.5)
  defp op(7, x), do: Nx.multiply(x, 0.5)
  defp op(8, x), do: Nx.negate(x)
  defp op(9, x), do: Nx.log1p(Nx.multiply(x, x))
  defp op(10, x), do: Nx.exp(Nx.negate(Nx.multiply(x, x)))
  defp op(11, x), do: Nx.subtract(x, Nx.sin(x))
end

defmodule Nx.Defn.CheckpointFuzzTest do
  @moduledoc """
  Property tests for `checkpoint/2`. A checkpoint changes how a function is
  evaluated, never what it computes, so every property builds a random
  program with checkpoints in random places and compares it against the same
  program with the checkpoints stripped.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Nx.Defn
  import Nx.Defn.Kernel, only: [checkpoint: 1, checkpoint: 2]
  import Nx.Testing

  alias Nx.Defn.CheckpointFuzzTest.Program
  import Program, only: [run: 2, strip: 1, step: 2]

  @fuzz_scale String.to_integer(System.get_env("FUZZ_SCALE", "1"))

  # ── Generators ───────────────────────────────────────────────────────────

  defp op_index, do: integer(0..(Program.op_count() - 1))

  defp program(depth) when depth <= 0, do: list_of(op_index(), min_length: 1, max_length: 4)

  defp program(depth) do
    list_of(
      frequency([
        {3, op_index()},
        {1, {:ckpt, program(depth - 1)}}
      ]),
      min_length: 1,
      max_length: 4
    )
  end

  defp tensor(shape, type \\ {:f, 64}, bound \\ 2.0) do
    count = Tuple.product(shape)

    bind(list_of(float(min: -bound, max: bound), length: count), fn vals ->
      constant(vals |> Nx.tensor(type: type) |> Nx.reshape(shape))
    end)
  end

  defp shape do
    frequency([
      {1, constant({})},
      {3, tuple({integer(1..5)})},
      {2, tuple({integer(1..3), integer(1..3)})}
    ])
  end

  defp loss(fun), do: fn x -> Nx.sum(fun.(x)) end

  defp assert_same_value_and_grad(with_ckpt, plain, x, opts \\ []) do
    {value, grad} = Nx.Defn.value_and_grad(x, loss(with_ckpt))
    {expected_value, expected_grad} = Nx.Defn.value_and_grad(x, loss(plain))
    assert_all_close(value, expected_value, opts)
    assert_all_close(grad, expected_grad, opts)
  end

  # ── Properties ───────────────────────────────────────────────────────────

  describe "segmentation invariance" do
    property "checkpoints anywhere in a chain leave value and gradient unchanged" do
      check all(
              steps <- program(3),
              shape <- shape(),
              x <- tensor(shape),
              max_runs: 60 * @fuzz_scale
            ) do
        assert_same_value_and_grad(&run(steps, &1), &run(strip(steps), &1), x)
      end
    end

    property "value and gradient agree to the last few bits for a single checkpoint" do
      check all(
              steps <- program(0),
              shape <- shape(),
              x <- tensor(shape),
              max_runs: 30 * @fuzz_scale
            ) do
        with_ckpt = fn x -> checkpoint(x, fn x -> run(steps, x) end) end
        {value, grad} = Nx.Defn.value_and_grad(x, loss(with_ckpt))
        {expected_value, expected_grad} = Nx.Defn.value_and_grad(x, loss(&run(steps, &1)))
        assert_all_close(value, expected_value, atol: 0, rtol: 1.0e-14)
        assert_all_close(grad, expected_grad, atol: 1.0e-15, rtol: 1.0e-14)
      end
    end
  end

  describe "fan-out" do
    property "an output used several times and an input also used directly" do
      check all(
              steps <- program(0),
              uses <- list_of(op_index(), min_length: 1, max_length: 4),
              direct? <- boolean(),
              shape <- shape(),
              x <- tensor(shape),
              max_runs: 30 * @fuzz_scale
            ) do
        consume = fn y, x ->
          zero = Nx.tensor(0.0, type: :f64)
          acc = Enum.reduce(uses, zero, fn i, acc -> Nx.add(acc, step(i, y)) end)
          if direct?, do: Nx.add(acc, Nx.multiply(x, y)), else: acc
        end

        with_ckpt = fn x -> consume.(checkpoint(x, fn x -> run(steps, x) end), x) end
        plain = fn x -> consume.(run(steps, x), x) end
        assert_same_value_and_grad(with_ckpt, plain, x)
      end
    end
  end

  describe "forms" do
    property "explicit, capture and zero-arity forms agree, for inputs and captures" do
      check all(
              steps <- program(0),
              shape <- shape(),
              x <- tensor(shape),
              w <- tensor(shape),
              max_runs: 30 * @fuzz_scale
            ) do
        body = fn x, w -> Nx.multiply(run(steps, x), w) end

        explicit = fn x -> checkpoint([x, w], body) end
        capture = fn x -> checkpoint(x, fn x -> body.(x, w) end) end
        zero = fn x -> checkpoint(fn -> body.(x, w) end) end
        plain = fn x -> body.(x, w) end

        for form <- [explicit, capture, zero] do
          assert_same_value_and_grad(form, plain, x)
        end

        expected_w = Nx.Defn.grad(w, fn w -> Nx.sum(body.(x, w)) end)
        capture_w = Nx.Defn.grad(w, fn w -> Nx.sum(checkpoint(x, fn x -> body.(x, w) end)) end)
        zero_w = Nx.Defn.grad(w, fn w -> Nx.sum(checkpoint(fn -> body.(x, w) end)) end)
        assert_all_close(capture_w, expected_w)
        assert_all_close(zero_w, expected_w)
      end
    end
  end

  describe "containers" do
    property "tuple and map inputs flow through a checkpoint" do
      check all(
              steps <- program(0),
              shape <- shape(),
              a <- tensor(shape),
              b <- tensor(shape),
              c <- tensor(shape),
              max_runs: 30 * @fuzz_scale
            ) do
        body = fn {p, %{q: q, r: r}} -> {run(steps, p), Nx.add(Nx.multiply(q, r), p)} end
        combine = fn {u, v} -> Nx.add(Nx.sum(u), Nx.sum(Nx.sin(v))) end

        with_ckpt = fn input -> combine.(checkpoint(input, body)) end
        plain = fn input -> combine.(body.(input)) end
        input = {a, %{q: b, r: c}}

        {value, grad} = Nx.Defn.value_and_grad(input, with_ckpt)
        {expected_value, expected_grad} = Nx.Defn.value_and_grad(input, plain)
        assert_all_close(value, expected_value)
        {ga, %{q: gb, r: gc}} = grad
        {ea, %{q: eb, r: ec}} = expected_grad
        assert_all_close(ga, ea)
        assert_all_close(gb, eb)
        assert_all_close(gc, ec)
      end
    end
  end

  describe "vectorized inputs" do
    property "a vectorized input gives the same vectorized value and gradient" do
      check all(
              steps <- program(2),
              batch <- integer(1..3),
              inner <- integer(1..4),
              x <- tensor({batch, inner}),
              max_runs: 30 * @fuzz_scale
            ) do
        vx = Nx.vectorize(x, :batch)
        {value, grad} = Nx.Defn.value_and_grad(vx, loss(&run(steps, &1)))
        {expected_value, expected_grad} = Nx.Defn.value_and_grad(vx, loss(&run(strip(steps), &1)))
        assert value.vectorized_axes == expected_value.vectorized_axes
        assert grad.vectorized_axes == expected_grad.vectorized_axes
        assert_all_close(value, expected_value)
        assert_all_close(grad, expected_grad)
      end
    end

    property "a reduction inside a checkpoint stays per batch entry" do
      check all(
              batch <- integer(1..3),
              inner <- integer(1..4),
              x <- tensor({batch, inner}),
              max_runs: 20 * @fuzz_scale
            ) do
        vx = Nx.vectorize(x, :batch)
        with_ckpt = fn x -> checkpoint(x, fn x -> Nx.sum(Nx.sin(x)) end) end
        plain = fn x -> Nx.sum(Nx.sin(x)) end

        value = Nx.Defn.jit_apply(with_ckpt, [vx])
        assert value.vectorized_axes == [batch: batch]
        assert_all_close(value, plain.(vx))
        assert_same_value_and_grad(with_ckpt, plain, vx)
      end
    end
  end

  describe "non-finite values" do
    property "the forward value is bit-identical to the plain function" do
      check all(
              steps <- program(2),
              x <- FuzzGen.bit_tensor(FuzzGen.shape(), {:f, 32}),
              max_runs: 40 * @fuzz_scale
            ) do
        plain = strip(steps)
        assert Nx.to_binary(run(steps, x)) == Nx.to_binary(run(plain, x))

        # Under the Evaluator a block body runs eagerly, so the tracer's
        # constant folding applies to the plain program only. Finite values
        # may differ in the last bit; NaN and infinity must agree exactly.
        jitted = Nx.Defn.jit_apply(&run(steps, &1), [x])
        reference = Nx.Defn.jit_apply(&run(plain, &1), [x])
        assert_equal(Nx.is_nan(jitted), Nx.is_nan(reference))
        assert_equal(Nx.is_infinity(jitted), Nx.is_infinity(reference))
        finite = Nx.logical_not(Nx.logical_or(Nx.is_nan(reference), Nx.is_infinity(reference)))

        assert_all_close(Nx.select(finite, jitted, 0), Nx.select(finite, reference, 0),
          atol: 0,
          rtol: 2.0e-7
        )
      end
    end
  end

  # Programs are passed through opts so defn can host the control flow, and
  # reach the plain Elixir helpers through transforms.

  deftransformp run_steps(steps, x), do: Program.run(steps, x)
  deftransformp apply_steps(x, steps, ckpt?), do: Program.apply_steps(x, steps, ckpt?)

  defn cond_inside(x, pivot, opts \\ []) do
    checkpoint(x, fn x ->
      if Nx.sum(x) > pivot,
        do: run_steps(opts[:steps], x),
        else: run_steps(opts[:other], x)
    end)
  end

  defn cond_around(x, pivot, opts \\ []) do
    if Nx.sum(x) > pivot,
      do: apply_steps(x, opts[:steps], opts[:ckpt]),
      else: run_steps(opts[:other], x)
  end

  defn loop(x, iterations, opts \\ []) do
    {_, acc} =
      while {i = 0, acc = x}, i < iterations do
        {i + 1, apply_steps(acc, opts[:steps], opts[:ckpt])}
      end

    acc
  end

  describe "control flow" do
    property "cond around and inside a checkpoint" do
      check all(
              steps <- program(1),
              other <- program(0),
              shape <- shape(),
              x <- tensor(shape),
              pivot <- float(min: -2.0, max: 2.0),
              max_runs: 30 * @fuzz_scale
            ) do
        plain = &cond_around(&1, pivot, steps: strip(steps), other: other, ckpt: false)

        assert_same_value_and_grad(&cond_inside(&1, pivot, steps: steps, other: other), plain, x)

        assert_same_value_and_grad(
          &cond_around(&1, pivot, steps: steps, other: other, ckpt: true),
          plain,
          x
        )
      end
    end

    property "checkpoint inside a while body" do
      # Values stay in [-1, 1] so repeated squaring cannot overflow f64,
      # which the BinaryBackend reports as an error rather than infinity.
      check all(
              steps <- program(0),
              iterations <- integer(0..4),
              shape <- shape(),
              x <- tensor(shape, {:f, 64}, 1.0),
              max_runs: 20 * @fuzz_scale
            ) do
        with_ckpt = &loop(&1, iterations, steps: steps, ckpt: true)
        plain = &loop(&1, iterations, steps: steps, ckpt: false)
        assert_same_value_and_grad(with_ckpt, plain, x)
      end
    end
  end

  describe "higher-order" do
    property "grad of grad through a checkpoint" do
      check all(
              steps <- program(1),
              n <- integer(1..4),
              x <- tensor({n}),
              max_runs: 20 * @fuzz_scale
            ) do
        second = fn fun -> fn x -> Nx.sum(Nx.Defn.grad(x, loss(fun))) end end
        grad = Nx.Defn.grad(x, second.(&run(steps, &1)))
        expected = Nx.Defn.grad(x, second.(&run(strip(steps), &1)))
        assert_all_close(grad, expected)
      end
    end
  end

  describe "equal shapes, different bodies" do
    property "several same-shaped checkpoints keep their own bodies" do
      check all(
              bodies <- list_of(program(0), min_length: 2, max_length: 4),
              shape <- shape(),
              x <- tensor(shape),
              max_runs: 30 * @fuzz_scale
            ) do
        chain = fn x, ckpt? ->
          Enum.reduce(bodies, x, fn steps, acc -> Program.apply_steps(acc, steps, ckpt?) end)
        end

        assert_same_value_and_grad(&chain.(&1, true), &chain.(&1, false), x)
      end
    end
  end

  describe "dtypes" do
    property "every float type gives the same value and gradient as plain" do
      check all(
              steps <- program(1),
              type <- member_of([{:f, 16}, {:bf, 16}, {:f, 32}, {:f, 64}]),
              shape <- shape(),
              x <- tensor(shape, type),
              max_runs: 30 * @fuzz_scale
            ) do
        {value, grad} = Nx.Defn.value_and_grad(x, loss(&run(steps, &1)))
        {expected_value, expected_grad} = Nx.Defn.value_and_grad(x, loss(&run(strip(steps), &1)))
        assert Nx.type(grad) == Nx.type(expected_grad)
        assert_equal(value, expected_value)

        # The two paths accumulate in a different order. The half types
        # cancel catastrophically in these chains, so only an absolute
        # bound is meaningful for them.
        if elem(type, 1) == 16 do
          assert_all_close(grad, expected_grad, atol: 1.0e-2, rtol: 0)
        else
          assert_all_close(grad, expected_grad, atol: 0, rtol: 1.0e-14)
        end
      end
    end
  end

  describe "argument errors" do
    property "a function whose arity does not match the inputs raises" do
      check all(
              inputs <- integer(1..3),
              extra <- integer(1..2),
              max_runs: 10
            ) do
        args = List.duplicate(Nx.tensor(1.0), inputs)
        fun = Function.capture(__MODULE__, :arity_fun, inputs + extra)

        assert_raise ArgumentError, ~r/expected a function of arity/, fn ->
          checkpoint(args, fun)
        end
      end
    end
  end

  for arity <- 2..5 do
    args = Macro.generate_arguments(arity, __MODULE__)
    def arity_fun(unquote_splicing(args)), do: hd([unquote_splicing(args)])
  end
end
