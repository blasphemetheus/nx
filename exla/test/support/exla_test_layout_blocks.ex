# Test-only block tags for the custom-call layout defaults
# (test/exla/mlir/custom_call_test.exs). They live here because protocols
# are consolidated before the test files compile.
defmodule EXLA.Test.DefaultLayoutBlock do
  @moduledoc false
  defstruct []
end

defmodule EXLA.Test.PinnedLayoutBlock do
  @moduledoc false
  defstruct []
end

defimpl EXLA.CustomCall, for: EXLA.Test.DefaultLayoutBlock do
  def call(_, _out, _args, _client),
    do: {:ok, %EXLA.CustomCall.Spec{call_target_name: "test_default_layouts"}}
end

defimpl EXLA.CustomCall, for: EXLA.Test.PinnedLayoutBlock do
  def call(_, _out, _args, _client) do
    {:ok,
     %EXLA.CustomCall.Spec{
       call_target_name: "test_pinned_layouts",
       mlir_attributes: [{"result_layouts", "[dense<[0, 1, 2]> : tensor<3xindex>]"}]
     }}
  end
end
