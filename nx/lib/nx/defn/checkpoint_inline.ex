defmodule Nx.Defn.CheckpointInline do
  @moduledoc false

  def expand(inputs, callback, caller) do
    arity = callback_arity!(callback)
    scope = MapSet.new(Macro.Env.vars(caller))

    {_, captures} =
      Macro.prewalk(callback, [], fn
        {name, _, context} = variable, captures when is_atom(name) and is_atom(context) ->
          if MapSet.member?(scope, {name, context}) do
            {variable, [variable | captures]}
          else
            {variable, captures}
          end

        node, captures ->
          {node, captures}
      end)

    captures =
      captures |> Enum.reverse() |> Enum.uniq_by(fn {name, _, context} -> {name, context} end)

    arguments = Macro.generate_unique_arguments(arity, __MODULE__)
    captured = {:{}, [], captures}

    pattern =
      {:{}, [],
       Enum.map(captures, fn {name, meta, context} ->
         {name, Keyword.put(meta, :generated, true), context}
       end)}

    quote do
      Nx.Defn.Kernel.__checkpoint_inline__(
        unquote(inputs),
        unquote(captured),
        fn {unquote_splicing(arguments)}, unquote(pattern) ->
          unquote(callback).(unquote_splicing(arguments))
        end
      )
    end
  end

  defp callback_arity!({:fn, _, [{:->, _, [arguments, _]} | _]}), do: length(arguments)
  defp callback_arity!({:&, _, [{:/, _, [_, arity]}]}) when is_integer(arity), do: arity

  defp callback_arity!({:&, _, [body]}) do
    {_, arity} =
      Macro.prewalk(body, 0, fn
        {:&, _, [position]} = node, arity when is_integer(position) ->
          {node, max(position, arity)}

        node, arity ->
          {node, arity}
      end)

    arity
  end

  defp callback_arity!(callback) do
    raise ArgumentError,
          "checkpoint_inline expects an inline fn or & callback, got: #{Macro.to_string(callback)}. " <>
            "Use checkpoint/2 with explicit inputs for a callback stored in a variable"
  end

  def run(inputs, captures, fun) do
    {_, tensors} =
      map_tensors(captures, [], fn tensor, tensors -> {tensor, [tensor | tensors]} end)

    tensors = tensors |> Enum.reverse() |> List.to_tuple()

    Nx.Defn.Kernel.checkpoint({List.to_tuple(List.wrap(inputs)), tensors}, fn {arguments, tensors} ->
      {captures, []} =
        map_tensors(captures, Tuple.to_list(tensors), fn _, [tensor | rest] -> {tensor, rest} end)

      fun.(arguments, captures)
    end)
  end

  defp map_tensors(%Nx.Tensor{} = tensor, acc, fun), do: fun.(tensor, acc)

  defp map_tensors(tuple, acc, fun) when is_tuple(tuple) do
    {values, acc} = tuple |> Tuple.to_list() |> Enum.map_reduce(acc, &map_tensors(&1, &2, fun))
    {List.to_tuple(values), acc}
  end

  defp map_tensors(map, acc, fun) when is_map(map),
    do: Nx.Container.traverse(map, acc, &map_tensors(&1, &2, fun))

  defp map_tensors(list, acc, fun) when is_list(list),
    do: Enum.map_reduce(list, acc, &map_tensors(&1, &2, fun))

  defp map_tensors(value, _acc, _fun) when is_function(value) do
    raise ArgumentError, "checkpoint_inline cannot discover tensors inside a captured function"
  end

  defp map_tensors(value, acc, _fun), do: {value, acc}
end
