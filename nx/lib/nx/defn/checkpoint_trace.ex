defmodule Nx.Defn.CheckpointTrace do
  @moduledoc false
  alias Nx.Defn.{Composite, Expr, Tree}

  @trace_key {__MODULE__, :created}

  def record(id) do
    case Process.get(@trace_key) do
      nil -> :ok
      created -> Process.put(@trace_key, MapSet.put(created, id))
    end

    id
  end

  def run(inputs, fun) do
    arguments = List.wrap(inputs)

    if Nx.Defn.Compiler.current() == nil do
      apply(fun, arguments)
    else
      {body, parameters, captures, context, axes} = trace(arguments, fun)
      ids = Enum.map(parameters ++ captures, & &1.data.id)

      Nx.Defn.Kernel.checkpoint(
        {List.to_tuple(arguments), List.to_tuple(captures)},
        fn {arguments, captures} ->
          values = Composite.flatten_list([arguments, captures]) |> Enum.map(&Nx.devectorize/1)
          body |> replay(ids, values, context) |> revectorize(axes)
        end
      )
    end
  end

  def replay(body, ids, values, context) do
    if Enum.all?(values, &match?(%Nx.Tensor{data: %Expr{}}, &1)) do
      instantiate(body, ids, values, context)
    else
      Nx.Defn.jit_apply(
        fn values -> instantiate(body, ids, Tuple.to_list(values), context) end,
        [List.to_tuple(values)],
        compiler: Nx.Defn.Evaluator
      )
    end
  end

  def trace(arguments, fun) do
    arguments = Composite.traverse(List.to_tuple(arguments), &Expr.tensor/1)
    tensors = Composite.flatten_list([arguments])
    context = Enum.find_value(tensors, :root, & &1.data.context)
    previous = Process.put(@trace_key, MapSet.new())

    try do
      {parameters, _} =
        Composite.traverse(arguments, 0, fn tensor, position ->
          parameter = Expr.parameter(Nx.devectorize(tensor), context, position)
          {Nx.vectorize(parameter, tensor.vectorized_axes), position + 1}
        end)

      result = apply(fun, Tuple.to_list(parameters))

      unless match?(%Nx.Tensor{}, result) or
               (is_tuple(result) and Enum.all?(Tuple.to_list(result), &match?(%Nx.Tensor{}, &1))) do
        raise ArgumentError, "checkpoint_capture expects a tensor or flat tuple of tensors"
      end

      axes = Composite.flatten_list([result]) |> Enum.map(& &1.vectorized_axes)
      body = Composite.traverse(result, &Nx.devectorize/1)
      created = Process.get(@trace_key)

      {_, {captures, _}} =
        Composite.traverse(body, {[], MapSet.new()}, &captures(&1, created, &2))

      {body, Composite.flatten_list([parameters]), Enum.reverse(captures), context, axes}
    after
      created = Process.get(@trace_key)

      if previous,
        do: Process.put(@trace_key, MapSet.union(previous, created)),
        else: Process.delete(@trace_key)
    end
  end

  defp captures(
         %Nx.Tensor{data: %Expr{op: :metadata, args: [_, %{custom_grad: _}]}},
         _created,
         _acc
       ) do
    raise ArgumentError,
          "checkpoint_capture prototype cannot convert custom_grad callbacks; use checkpoint/2 with explicit inputs"
  end

  defp captures(
         %Nx.Tensor{data: %Expr{id: id, context: context}} = tensor,
         created,
         {captures, seen}
       ) do
    cond do
      MapSet.member?(seen, id) ->
        {tensor, {captures, seen}}

      context == nil ->
        {tensor, {captures, MapSet.put(seen, id)}}

      MapSet.member?(created, id) ->
        {_, acc} =
          Tree.apply_args(
            tensor,
            :scope,
            {captures, MapSet.put(seen, id)},
            &captures(&1, created, &2)
          )

        {tensor, acc}

      true ->
        {tensor, {[Nx.devectorize(tensor) | captures], MapSet.put(seen, id)}}
    end
  end

  def instantiate(body, ids, values, context) do
    substitutions = Map.new(Enum.zip(ids, values))
    target_context = Enum.find_value(values, context, & &1.data.context)

    {result, _} =
      Composite.traverse(
        body,
        %{},
        &instantiate_node(&1, substitutions, context, target_context, &2)
      )

    result
  end

  defp instantiate_node(
         %Nx.Tensor{data: %Expr{op: :metadata, args: [_, %{custom_grad: _}]}},
         _substitutions,
         _source,
         _target,
         _cache
       ) do
    raise ArgumentError,
          "checkpoint_capture prototype cannot convert custom_grad callbacks; use checkpoint/2 with explicit inputs"
  end

  defp instantiate_node(
         %Nx.Tensor{data: %Expr{id: id, context: context}} = tensor,
         substitutions,
         source,
         target,
         cache
       ) do
    case {substitutions, cache} do
      {%{^id => replacement}, _} ->
        {replacement, cache}

      {_, %{^id => replacement}} ->
        {replacement, cache}

      _ ->
        {args, cache} =
          Tree.apply_args(
            tensor,
            :all,
            cache,
            &instantiate_node(&1, substitutions, source, target, &2)
          )

        result = Tree.put_args(tensor, args)
        result = if context == source, do: put_in(result.data.context, target), else: result
        {result, Map.put(cache, id, result)}
    end
  end

  def revectorize(body, axes) do
    {result, []} =
      Composite.traverse(body, axes, fn tensor, [axes | rest] ->
        {Nx.vectorize(tensor, axes), rest}
      end)

    result
  end
end
