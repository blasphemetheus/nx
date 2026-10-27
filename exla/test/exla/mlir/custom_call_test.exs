defmodule EXLA.MLIR.CustomCallTest do
  use EXLA.Case, async: true

  alias EXLA.MLIR.Function
  alias EXLA.MLIR.Module
  alias EXLA.MLIR.Value

  # The CPU custom call is more precise than the generic QR used on other platforms
  # f16 machine epsilon is ~9.77e-4; atol must be above that
  defp tol_opts({:f, 16}, _platform), do: [atol: 7.0e-3, rtol: 7.0e-3]
  defp tol_opts({:bf, 16}, :host), do: [atol: 6.0e-2, rtol: 6.0e-2]
  defp tol_opts({:bf, 16}, _platform), do: [atol: 2.5e-1, rtol: 2.5e-1]
  defp tol_opts({:f, 64}, _platform), do: [atol: 1.0e-14, rtol: 1.0e-15]
  defp tol_opts({:f, 32}, :host), do: [atol: 1.0e-6, rtol: 1.0e-6]
  defp tol_opts({:f, 32}, _platform), do: [atol: 1.0e-5, rtol: 1.0e-5]

  describe "qr" do
    for type <- [bf: 16, f: 16, f: 32, f: 64] do
      test "works for input type #{inspect(type)}" do
        tol_opts = tol_opts(unquote(type), EXLAHelpers.client().platform)

        square = Nx.iota({4, 4}, type: unquote(type))
        tall = Nx.iota({4, 3}, type: unquote(type))
        wide = Nx.iota({3, 4}, type: unquote(type))

        fun =
          EXLA.jit(fn t ->
            {q, r} = Nx.LinAlg.qr(t, mode: :reduced)
            Nx.dot(q, r)
          end)

        assert_all_close(fun.(square), square, tol_opts)
        assert_all_close(fun.(tall), tall, tol_opts)
        assert_all_close(fun.(wide), wide, tol_opts)
      end
    end
  end

  test "MLIR attribute names cannot override or duplicate attributes" do
    typespec = EXLA.Typespec.tensor({:f, 32}, {1})

    for mlir_attributes <- [
          [{"call_target_name", ~s("replacement")}],
          [{"test.attribute", "1 : i64"}, {"test.attribute", "2 : i64"}]
        ] do
      assert_raise ArgumentError, ~r/must be unique and cannot override/, fn ->
        Module.new([typespec], [typespec], fn function ->
          [argument] = Function.get_arguments(function)
          Value.custom_call([argument], [typespec], "original", [], mlir_attributes)
        end)
      end
    end
  end

  test "string MLIR attribute names preserve backend config without creating atoms" do
    typespec = EXLA.Typespec.tensor({:f, 32}, {1})
    name = "test.attribute_#{System.unique_integer([:positive])}"

    assert_raise ArgumentError, fn -> String.to_existing_atom(name) end

    mlir =
      Module.new([typespec], [typespec], fn function ->
        [argument] = Function.get_arguments(function)

        [result] =
          Value.custom_call(
            [argument],
            [typespec],
            "original",
            [{"test_backend", "42 : i64"}],
            [{name, "1 : i64"}]
          )

        Value.func_return(function, [result])
        Module.as_string(function.module)
      end)

    assert mlir =~ "backend_config = {test_backend = 42 : i64}"
    assert mlir =~ "#{name} = 1 : i64"
    assert_raise ArgumentError, fn -> String.to_existing_atom(name) end
  end

  describe "layouts on protocol custom calls" do
    # A handler that reads row-major gets a buffer in whatever layout XLA
    # found convenient unless the op pins its layouts. The lowering pins
    # row-major by default; a Spec that sets either attribute wins. The
    # block tags and their protocol impls are in test/support.
    defmodule Fun do
      import Nx.Defn

      defn default_layouts(x, w) do
        out = Nx.template({2, 4, 3}, {:f, 32})
        Nx.block(%EXLA.Test.DefaultLayoutBlock{}, [x, w], out, fn _, x, w -> Nx.dot(x, w) end)
      end

      defn pinned_layouts(x, w) do
        out = Nx.template({2, 4, 3}, {:f, 32})
        Nx.block(%EXLA.Test.PinnedLayoutBlock{}, [x, w], out, fn _, x, w -> Nx.dot(x, w) end)
      end
    end

    defp mlir_of(fun, args) do
      try do
        Nx.Defn.jit_apply(fun, args, compiler: EXLA, module_compilation: :to_mlir)
      catch
        :throw, {:mlir_module, ref, _used_inputs, _output} ->
          EXLA.MLIR.Module.as_string(%EXLA.MLIR.Module{ref: ref})
      end
    end

    test "row-major operand and result layouts are emitted by default" do
      x = Nx.iota({2, 4, 5}, type: {:f, 32})
      w = Nx.iota({5, 3}, type: {:f, 32})
      mlir = mlir_of(&Fun.default_layouts/2, [x, w])

      assert mlir =~ "stablehlo.custom_call @test_default_layouts("

      assert mlir =~
               "operand_layouts = [dense<[2, 1, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>]"

      assert mlir =~ "result_layouts = [dense<[2, 1, 0]> : tensor<3xindex>]"
    end

    test "layouts set by the Spec are kept and not duplicated" do
      x = Nx.iota({2, 4, 5}, type: {:f, 32})
      w = Nx.iota({5, 3}, type: {:f, 32})
      mlir = mlir_of(&Fun.pinned_layouts/2, [x, w])

      assert mlir =~ "result_layouts = [dense<[0, 1, 2]> : tensor<3xindex>]"
      refute mlir =~ "operand_layouts"
    end
  end
end
