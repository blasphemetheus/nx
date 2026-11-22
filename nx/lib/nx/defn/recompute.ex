defmodule Nx.Defn.Recompute do
  @moduledoc false
  alias Nx.Defn.{Expr, Tree}

  # Repeating effects or opaque callbacks needs a separate contract. Keep
  # the experiment limited to a straight-line, single-output tensor recipe.
  def validate!(%Nx.Tensor{} = tensor) do
    visit(tensor, MapSet.new())
    tensor
  end

  def validate!(_), do: raise(ArgumentError, "recompute_each_use expects one tensor output")

  defp visit(%Nx.Tensor{data: %Expr{id: id, op: op, args: args}} = tensor, seen) do
    if op in [:block, :cond, :while, :io_call, :runtime_call] or
         (op == :metadata and match?([_, %{custom_grad: _}], args)) do
      raise ArgumentError, "recompute_each_use prototype does not support #{op} in its body"
    end

    if MapSet.member?(seen, id) do
      {tensor, seen}
    else
      {_, seen} = Tree.apply_args(tensor, :all, MapSet.put(seen, id), &visit/2)
      {tensor, seen}
    end
  end

  defp visit(tensor, seen), do: {tensor, seen}
end
