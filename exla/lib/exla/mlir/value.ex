defmodule EXLA.MLIR.Value do
  @moduledoc false
  # Representation of an MLIR Value.
  #
  # MLIR Values are SSA and generally are either operations or
  # block arguments. This module is used to construct most of the
  # MLIR operations.
  #
  # See the full specification of the stablehlo MLIR dialect [1]. Note
  # that the URL points to the exact stablehlo revision that we depend
  # on via elixir-nx/xla.
  #
  # [1]: https://github.com/openxla/stablehlo/blob/04291aea6b50d9573e6f4de184938d83b9564cd0/docs/spec.md

  defstruct [:ref, :function]

  alias __MODULE__
  alias EXLA.Typespec
  alias EXLA.MLIR.Region
  alias EXLA.MLIR.Function

  @bin_ops %{
    add: "stablehlo.add",
    subtract: "stablehlo.subtract",
    multiply: "stablehlo.multiply",
    divide: "stablehlo.divide",
    pow: "stablehlo.power",
    min: "stablehlo.minimum",
    max: "stablehlo.maximum",
    remainder: "stablehlo.remainder",
    atan2: "stablehlo.atan2",
    bitwise_and: "stablehlo.and",
    bitwise_or: "stablehlo.or",
    bitwise_xor: "stablehlo.xor",
    left_shift: "stablehlo.shift_left",
    right_shift_arithmetic: "stablehlo.shift_right_arithmetic",
    right_shift_logical: "stablehlo.shift_right_logical"
  }

  for {op, op_name} <- @bin_ops do
    def unquote(op)(%Value{function: func} = lhs, %Value{function: func} = rhs, typespec) do
      result_types = typespecs_to_mlir_types([typespec])
      op(func, unquote(op_name), [lhs, rhs], result_types) |> one!()
    end
  end

  @bin_comparison_ops %{
    equal: :eq,
    less: :lt,
    less_equal: :le,
    greater: :gt,
    greater_equal: :ge,
    not_equal: :ne
  }

  for {op, direction} <- @bin_comparison_ops do
    def unquote(op)(
          %Value{function: func} = lhs,
          %Value{function: func} = rhs,
          typespec,
          opts \\ []
        ) do
      compare_and_return_bool(func, lhs, rhs, typespec, unquote(direction), opts[:total_order])
    end
  end

  defp compare_and_return_bool(func, lhs, rhs, typespec, direction, total_order? \\ false) do
    %{type: lhs_type} = get_typespec(lhs)
    %{type: rhs_type} = get_typespec(rhs)

    comparison_type =
      cond do
        Nx.Type.complex?(lhs_type) or Nx.Type.complex?(rhs_type) ->
          [compare_type: attr_comparison_type(:float)]

        Nx.Type.float?(lhs_type) or Nx.Type.float?(rhs_type) ->
          attr =
            if total_order? do
              attr_comparison_type(:totalorder)
            else
              attr_comparison_type(:float)
            end

          [compare_type: attr]

        true ->
          []
      end

    attributes = [comparison_direction: attr_comparison_direction(direction)] ++ comparison_type

    result_types = typespecs_to_mlir_types([Typespec.to_type(typespec, {:pred, 8})])

    op(func, "stablehlo.compare", [lhs, rhs], result_types, attributes: attributes) |> one!()
  end

  @unary_ops %{
    abs: "stablehlo.abs",
    exp: "stablehlo.exponential",
    expm1: "stablehlo.exponential_minus_one",
    floor: "stablehlo.floor",
    ceil: "stablehlo.ceil",
    round: "stablehlo.round_nearest_afz",
    log: "stablehlo.log",
    log1p: "stablehlo.log_plus_one",
    sigmoid: "stablehlo.logistic",
    sign: "stablehlo.sign",
    cos: "stablehlo.cosine",
    sin: "stablehlo.sine",
    tan: "chlo.tan",
    acos: "chlo.acos",
    asin: "chlo.asin",
    atan: "chlo.atan",
    cosh: "chlo.cosh",
    sinh: "chlo.sinh",
    tanh: "stablehlo.tanh",
    acosh: "chlo.acosh",
    asinh: "chlo.asinh",
    atanh: "chlo.atanh",
    sqrt: "stablehlo.sqrt",
    cbrt: "stablehlo.cbrt",
    bitwise_not: "stablehlo.not",
    erf: "chlo.erf",
    erfc: "chlo.erfc",
    erf_inv: "chlo.erf_inv",
    rsqrt: "stablehlo.rsqrt",
    negate: "stablehlo.negate",
    count_leading_zeros: "stablehlo.count_leading_zeros",
    population_count: "stablehlo.popcnt",
    real: "stablehlo.real",
    imag: "stablehlo.imag",
    conjugate: "chlo.conj"
  }

  for {op, op_name} <- @unary_ops do
    def unquote(op)(%Value{function: func} = operand, typespec) do
      result_types = typespecs_to_mlir_types([typespec])
      op(func, unquote(op_name), [operand], result_types, []) |> one!()
    end
  end

  def is_infinity(%Value{function: func} = operand, out_typespec) do
    %{type: type} = get_typespec(operand)

    typespec = Typespec.to_type(out_typespec, {:pred, 8})

    result =
      cond do
        Nx.Type.complex?(type) ->
          float_typespec = Typespec.to_type(typespec, complex_part_type(type))
          real = real(operand, float_typespec)
          imag = imag(operand, float_typespec)
          is_inf_real = is_infinity(real, typespec)
          is_inf_imag = is_infinity(imag, typespec)
          bitwise_or(is_inf_real, is_inf_imag, typespec)

        Nx.Type.integer?(type) ->
          # Integers are never infinity. We use inequality to make sure
          # the operand is still a part of the computation
          not_equal(operand, operand, typespec)

        true ->
          result_types = typespecs_to_mlir_types([typespec])
          op(func, "chlo.is_inf", [operand], result_types) |> one!()
      end

    if out_typespec.type == typespec.type do
      result
    else
      convert(result, out_typespec)
    end
  end

  def is_nan(%Value{} = operand, out_typespec) do
    typespec = Typespec.to_type(out_typespec, {:pred, 8})

    # Only NaN is not equal to itself
    result = not_equal(operand, operand, typespec)

    if out_typespec.type == typespec.type do
      result
    else
      convert(result, out_typespec)
    end
  end

  def reshape(%Value{function: func} = operand, typespec) do
    result_types = typespecs_to_mlir_types([typespec])
    op(func, "stablehlo.reshape", [operand], result_types) |> one!()
  end

  def reverse(%Value{function: func} = operand, dims, typespec) do
    result_types = typespecs_to_mlir_types([typespec])
    attributes = [dimensions: attr_array_i64_elements(dims)]
    op(func, "stablehlo.reverse", [operand], result_types, attributes: attributes) |> one!()
  end

  def transpose(%Value{function: func} = operand, axes, typespec) do
    result_types = typespecs_to_mlir_types([typespec])
    attributes = [permutation: attr_array_i64_elements(axes)]
    op(func, "stablehlo.transpose", [operand], result_types, attributes: attributes) |> one!()
  end

  def slice(%Value{function: func} = operand, starts, limits, strides, typespec) do
    result_types = typespecs_to_mlir_types([typespec])

    attributes = [
      start_indices: attr_array_i64_elements(starts),
      limit_indices: attr_array_i64_elements(limits),
      strides: attr_array_i64_elements(strides)
    ]

    op(func, "stablehlo.slice", [operand], result_types, attributes: attributes) |> one!()
  end

  def dynamic_slice(%Value{function: func} = operand, starts, lengths, typespec) do
    result_types = typespecs_to_mlir_types([typespec])
    operands = [operand] ++ starts
    attributes = [slice_sizes: attr_array_i64_elements(lengths)]
    op(func, "stablehlo.dynamic_slice", operands, result_types, attributes: attributes) |> one!()
  end

  def convert(%Value{function: func} = operand, typespec) do
    result_types = typespecs_to_mlir_types([typespec])
    op(func, "stablehlo.convert", [operand], result_types) |> one!()
  end

  def bitcast_convert(%Value{function: func} = operand, typespec) do
    result_types = typespecs_to_mlir_types([typespec])
    op(func, "stablehlo.bitcast_convert", [operand], result_types) |> one!()
  end

  def top_k(%Value{function: func} = operand, k, typespecs) do
    [typespec, index_typespec] = typespecs
    result_types = typespecs_to_mlir_types([typespec, Typespec.to_type(index_typespec, {:s, 32})])

    attributes = [k: attr_i64(k)]
    [result, idx] = op(func, "chlo.top_k", [operand], result_types, attributes: attributes)

    idx = convert(idx, index_typespec)

    [result, idx]
  end

  def sort(
        [%Value{function: func} | _] = operands,
        %Region{ref: comparator},
        axis,
        stable,
        typespecs
      )
      when is_integer(axis) and is_boolean(stable) do
    result_types = typespecs_to_mlir_types(typespecs)

    attributes = [
      dimension: attr_i64(axis),
      is_stable: attr_boolean(stable)
    ]

    regions = [comparator]

    op(func, "stablehlo.sort", operands, result_types, attributes: attributes, regions: regions)
  end

  def iota(%Function{} = func, dim, typespec) do
    result_types = typespecs_to_mlir_types([typespec])
    attributes = [iota_dimension: attr_i64(dim)]
    op(func, "stablehlo.iota", [], result_types, attributes: attributes) |> one!()
  end

  def constant(%Function{} = func, data, typespec) do
    result_types = typespecs_to_mlir_types([typespec])
    value = attr_dense_elements(data, typespec.type, typespec.shape)
    attributes = [value: value]
    op(func, "stablehlo.constant", [], result_types, attributes: attributes) |> one!()
  end

  def dot_general(
        %Value{function: func} = lhs,
        %Value{function: func} = rhs,
        dnums,
        precision_config,
        typespec
      ) do
    result_types = typespecs_to_mlir_types([typespec])

    attr_precision_config = attr_precision_config(precision_config)

    {contract_axes1, batch_axes1, contract_axes2, batch_axes2} = dnums

    dot_dimension_numbers =
      attr_struct("stablehlo.dot",
        lhs_batching_dimensions: join_list(batch_axes1),
        rhs_batching_dimensions: join_list(batch_axes2),
        lhs_contracting_dimensions: join_list(contract_axes1),
        rhs_contracting_dimensions: join_list(contract_axes2)
      )

    attributes = [
      dot_dimension_numbers: dot_dimension_numbers,
      precision_config: "[#{attr_precision_config}, #{attr_precision_config}]"
    ]

    op(func, "stablehlo.dot_general", [lhs, rhs], result_types, attributes: attributes) |> one!()
  end

  def broadcast_in_dim(%Value{function: func} = operand, axes, typespec) do
    result_types = typespecs_to_mlir_types([typespec])

    attributes = [
      broadcast_dimensions: attr_array_i64_elements(axes)
    ]

    op(func, "stablehlo.broadcast_in_dim", [operand], result_types, attributes: attributes)
    |> one!()
  end

  def concatenate([%Value{function: func} | _rest] = operands, dimension, typespec) do
    result_types = typespecs_to_mlir_types([typespec])
    attributes = [dimension: attr_i64(dimension)]
    op(func, "stablehlo.concatenate", operands, result_types, attributes: attributes) |> one!()
  end

  def clamp(
        %Value{function: func} = operand,
        %Value{function: func} = min,
        %Value{function: func} = max,
        typespec
      ) do
    result_types = typespecs_to_mlir_types([typespec])
    op(func, "stablehlo.clamp", [min, operand, max], result_types) |> one!()
  end

  def select(
        %Value{function: func} = pred,
        %Value{function: func} = on_true,
        %Value{function: func} = on_false,
        typespec
      ) do
    result_types = typespecs_to_mlir_types([typespec])
    op(func, "stablehlo.select", [pred, on_true, on_false], result_types) |> one!()
  end

  def pad(
        %Value{function: func} = operand,
        %Value{function: func} = pad,
        padding_config,
        typespec
      ) do
    result_types = typespecs_to_mlir_types([typespec])

    {padding_low, padding_high, padding_mid} = unzip_padding_config(padding_config)

    attributes = [
      edge_padding_low: attr_array_i64_elements(padding_low),
      edge_padding_high: attr_array_i64_elements(padding_high),
      interior_padding: attr_array_i64_elements(padding_mid)
    ]

    op(func, "stablehlo.pad", [operand, pad], result_types, attributes: attributes) |> one!()
  end

  defp unzip_padding_config(padding_config),
    do: unzip_padding_config(padding_config, {[], [], []})

  defp unzip_padding_config([], {low_acc, high_acc, mid_acc}) do
    {Enum.reverse(low_acc), Enum.reverse(high_acc), Enum.reverse(mid_acc)}
  end

  defp unzip_padding_config([{low, high, mid} | rest], {low_acc, high_acc, mid_acc}) do
    unzip_padding_config(rest, {[low | low_acc], [high | high_acc], [mid | mid_acc]})
  end

  def fft(%Value{function: func} = value, fft_kind, fft_length, typespec)
      when fft_kind in [:fft, :ifft]
      when is_list(fft_length) or is_integer(fft_length) do
    result_types = typespecs_to_mlir_types([typespec])

    fft_type = attr_fft_type(fft_kind)

    attributes = [
      fft_type: fft_type,
      fft_length: attr_array_i64_elements(List.wrap(fft_length))
    ]

    op(func, "stablehlo.fft", [value], result_types, attributes: attributes) |> one!()
  end

  def scatter(
        %Value{function: func} = target,
        %Value{function: func} = indices,
        %Value{function: func} = updates,
        kind,
        indices_rank,
        update_window_dims,
        inserted_window_dims,
        index_dims_to_window_dims,
        typespec
      )
      when kind in [:add, :put] and is_integer(indices_rank) and is_list(update_window_dims) and
             is_list(inserted_window_dims) and is_list(index_dims_to_window_dims) do
    result_types = typespecs_to_mlir_types([typespec])

    operands = [target, indices, updates]

    scatter_dimension_numbers =
      attr_struct("stablehlo.scatter",
        update_window_dims: join_list(update_window_dims),
        inserted_window_dims: join_list(inserted_window_dims),
        scatter_dims_to_operand_dims: join_list(index_dims_to_window_dims),
        index_vector_dim: Integer.to_string(indices_rank)
      )

    attributes = [scatter_dimension_numbers: scatter_dimension_numbers]

    scatter_computation = scatter_computation(func, kind, typespec)
    regions = [scatter_computation.ref]

    op(func, "stablehlo.scatter", operands, result_types,
      attributes: attributes,
      regions: regions
    )
    |> one!()
  end

  defp scatter_computation(%Function{} = function, kind, typespec) do
    arg_typespec = Typespec.to_shape(typespec, {})
    {region, [value, update]} = Function.push_region(function, [arg_typespec, arg_typespec])

    res =
      case kind do
        :add -> add(value, update, arg_typespec)
        :put -> update
      end

    return(function, [res])

    Function.pop_region(function)

    region
  end

  def select_and_scatter(
        %Value{function: func} = target,
        %Value{function: func} = source,
        %Value{function: func} = init_value,
        comparison,
        window_dimensions,
        window_strides,
        padding,
        typespec
      )
      when comparison in [:gt, :lt] do
    operands = [target, source, init_value]

    result_types = typespecs_to_mlir_types([typespec])

    attributes = [
      window_dimensions: attr_array_i64_elements(window_dimensions),
      window_strides: attr_array_i64_elements(window_strides),
      padding: attr_padding(padding)
    ]

    select_computation = select_computation(func, comparison, typespec)
    scatter_computation = scatter_computation(func, :add, typespec)
    regions = [select_computation.ref, scatter_computation.ref]

    op(func, "stablehlo.select_and_scatter", operands, result_types,
      attributes: attributes,
      regions: regions
    )
    |> one!()
  end

  defp select_computation(function, direction, typespec) do
    arg_typespec = Typespec.to_shape(typespec, {})
    {region, [arg0, arg1]} = Function.push_region(function, [arg_typespec, arg_typespec])

    res = compare_and_return_bool(function, arg0, arg1, arg_typespec, direction)
    return(function, [res])

    Function.pop_region(function)

    region
  end

  def gather(
        %Value{function: func} = source,
        %Value{function: func} = indices,
        index_vector_dim,
        slice_sizes,
        offset_dims,
        collapsed_slice_dims,
        start_index_map,
        typespec
      ) do
    result_types = typespecs_to_mlir_types([typespec])

    dimension_numbers =
      attr_struct("stablehlo.gather",
        offset_dims: join_list(offset_dims),
        collapsed_slice_dims: join_list(collapsed_slice_dims),
        start_index_map: join_list(start_index_map),
        index_vector_dim: Integer.to_string(index_vector_dim)
      )

    attributes = [
      dimension_numbers: dimension_numbers,
      slice_sizes: attr_array_i64_elements(slice_sizes),
      indices_are_sorted: attr_boolean(false)
    ]

    op(func, "stablehlo.gather", [source, indices], result_types, attributes: attributes)
    |> one!()
  end

  defp attr_precision_config(precision_config) do
    case precision_config do
      :default ->
        attr_precision(:default)

      :high ->
        attr_precision(:high)

      :highest ->
        attr_precision(:highest)

      _ ->
        raise ArgumentError,
              "expected precision configuration to be one of" <>
                " :default, :high, or :highest," <>
                " got: #{inspect(precision_config)}"
    end
  end

  def convolution(
        %Value{function: func} = tensor,
        %Value{function: func} = kernel,
        strides,
        padding,
        input_dilation,
        kernel_dilation,
        dimension_numbers,
        feature_group_count,
        batch_group_count,
        precision_config,
        typespec
      ) do
    result_types = typespecs_to_mlir_types([typespec])

    attr_precision_config = attr_precision_config(precision_config)

    attributes = [
      window_strides: attr_array_i64_elements(strides),
      padding: attr_padding(padding),
      lhs_dilation: attr_array_i64_elements(input_dilation),
      rhs_dilation: attr_array_i64_elements(kernel_dilation),
      dimension_numbers: attr_conv_dimension_numbers(dimension_numbers),
      feature_group_count: attr_i64(feature_group_count),
      batch_group_count: attr_i64(batch_group_count),
      precision_config: "[#{attr_precision_config}, #{attr_precision_config}]"
    ]

    op(func, "stablehlo.convolution", [tensor, kernel], result_types, attributes: attributes)
    |> one!()
  end

  defp attr_conv_dimension_numbers(dimension_numbers) do
    {input_permutation, kernel_permutation, output_permutation} = dimension_numbers
    input_string = convolution_dims_permutation(input_permutation, "b", "f")
    kernel_string = convolution_dims_permutation(kernel_permutation, "o", "i")
    output_string = convolution_dims_permutation(output_permutation, "b", "f")
    "#stablehlo.conv<[#{input_string}]x[#{kernel_string}]->[#{output_string}]>"
  end

  defp convolution_dims_permutation(permutation, dim1_mark, dim2_mark) do
    [dim1, dim2 | spatial_dims] = permutation

    dims_with_marks =
      [{dim1, dim1_mark}, {dim2, dim2_mark}] ++
        Enum.with_index(spatial_dims, fn dim, idx -> {dim, Integer.to_string(idx)} end)

    dims_with_marks
    |> Enum.sort()
    |> Enum.map_join(",", fn {_dim, mark} -> mark end)
  end

  def triangular_solve(
        %Value{function: func} = a,
        %Value{function: func} = b,
        left_side,
        lower,
        transform,
        typespec
      ) do
    result_types = typespecs_to_mlir_types([typespec])

    complex? = Nx.Type.complex?(typespec.type)

    transpose_a =
      case transform do
        :transpose when complex? -> attr_transpose(:adjoint)
        :transpose -> attr_transpose(:transpose)
        :none -> attr_transpose(:no_transpose)
      end

    attributes = [
      left_side: attr_boolean(left_side),
      lower: attr_boolean(lower),
      unit_diagonal: attr_boolean(false),
      transpose_a: transpose_a
    ]

    op(func, "stablehlo.triangular_solve", [a, b], result_types, attributes: attributes) |> one!()
  end

  def dynamic_update_slice(%Value{function: func} = operand, updates, starts, typespec) do
    result_types = typespecs_to_mlir_types([typespec])

    op(func, "stablehlo.dynamic_update_slice", [operand, updates] ++ starts, result_types)
    |> one!()
  end

  def reduce(
        %Region{ref: reducer},
        [%Value{function: func} | _] = init_values,
        [%Value{function: func} | _] = inputs,
        dimensions,
        typespecs
      ) do
    operands = inputs ++ init_values
    result_types = typespecs_to_mlir_types(typespecs)
    attributes = [dimensions: attr_array_i64_elements(dimensions)]
    regions = [reducer]
    op(func, "stablehlo.reduce", operands, result_types, attributes: attributes, regions: regions)
  end

  def window_reduce(
        %Region{ref: reducer},
        [%Value{function: func} | _] = init_values,
        [%Value{function: func} | _] = inputs,
        window_dimensions,
        window_strides,
        input_dilations,
        window_dilations,
        padding,
        typespecs
      ) do
    operands = inputs ++ init_values
    result_types = typespecs_to_mlir_types(typespecs)

    attributes = [
      window_dimensions: attr_array_i64_elements(window_dimensions),
      window_strides: attr_array_i64_elements(window_strides),
      base_dilations: attr_array_i64_elements(input_dilations),
      window_dilations: attr_array_i64_elements(window_dilations),
      padding: attr_padding(padding)
    ]

    regions = [reducer]

    op(func, "stablehlo.reduce_window", operands, result_types,
      attributes: attributes,
      regions: regions
    )
  end

  def if_op(
        %Value{function: func} = pred,
        %Region{ref: on_true},
        %Region{ref: on_false},
        typespecs
      ) do
    result_types = typespecs_to_mlir_types(typespecs)
    regions = [on_true, on_false]
    pred = convert(pred, Typespec.tensor({:pred, 8}, {}))
    op(func, "stablehlo.if", [pred], result_types, regions: regions)
  end

  def infeed(%Value{function: func} = token, typespecs) do
    result_types = typespecs_to_mlir_types(typespecs ++ [Typespec.token()])
    results = op(func, "stablehlo.infeed", [token], result_types)
    {results, [token]} = Enum.split(results, -1)
    {token, results}
  end

  def outfeed(%Value{} = input, token), do: outfeed([input], token)

  def outfeed(inputs, %Value{function: func} = token) do
    result_types = [type_token()]
    op(func, "stablehlo.outfeed", inputs ++ [token], result_types) |> one!()
  end

  def create_token(%Function{} = func) do
    result_types = [type_token()]
    op(func, "stablehlo.create_token", [], result_types) |> one!()
  end

  def call(%Function{} = func, args, %Function{} = computation, typespecs) do
    result_types = typespecs_to_mlir_types(typespecs)
    attributes = [callee: attr_symbol_reference(computation.name)]
    op(func, "func.call", args, result_types, attributes: attributes)
  end

  def while(%Function{} = func, %Region{ref: pred}, %Region{ref: body}, initial) do
    typespecs = Enum.map(initial, &get_typespec/1)
    result_types = typespecs_to_mlir_types(typespecs)

    regions = [pred, body]

    op(func, "stablehlo.while", initial, result_types, regions: regions)
  end

  def func_return(func, values) when is_list(values) do
    op(func, "func.return", values, [])
  end

  def return(func, values) when is_list(values) do
    op(func, "stablehlo.return", values, [])
  end

  def eigh(%Value{function: func} = value, eigenvals_typespec, eigenvecs_typespec) do
    %{type: op_type} = get_typespec(value)

    operands = [value]
    result_types = typespecs_to_mlir_types([eigenvals_typespec, eigenvecs_typespec])

    call_target_name =
      case op_type do
        {:f, 32} ->
          "eigh_cpu_custom_call_f32"

        {:f, 64} ->
          "eigh_cpu_custom_call_f64"

        type ->
          # Due to matching on EXLA.Defn, we are sure that the device here is always :host
          raise "Eigh decomposition not supported on :host device for type #{inspect(type)}"
      end

    attributes = [
      call_target_name: attr_string(call_target_name),
      api_version: attr_i32(4)
    ]

    [eigenvals, eigenvecs] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {eigenvals, eigenvecs}
  end

  def qr(%Value{function: func} = value, q_typespec, r_typespec) do
    %{type: op_type} = get_typespec(value)

    operands = [value]
    result_types = typespecs_to_mlir_types([q_typespec, r_typespec])

    call_target_name =
      case op_type do
        {:f, 32} ->
          "qr_cpu_custom_call_f32"

        {:f, 64} ->
          "qr_cpu_custom_call_f64"

        {:f, 16} ->
          "qr_cpu_custom_call_f16"

        {:bf, 16} ->
          "qr_cpu_custom_call_bf16"

        type ->
          # Due to matching on EXLA.Defn, we are sure that the device here is always :host
          raise "QR decomposition not supported on :host device for type #{inspect(type)}"
      end

    attributes = [
      call_target_name: attr_string(call_target_name),
      api_version: attr_i32(4)
    ]

    [q, r] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {q, r}
  end

  def lu(%Value{function: func} = value, p_typespec, l_typespec, u_typespec) do
    %{type: op_type} = get_typespec(value)

    operands = [value]

    # Force P to always be u8 to avoid requiring too many template instances during custom_call registration
    u8_typespec = Typespec.to_type(p_typespec, {:u, 8})
    result_types = typespecs_to_mlir_types([u8_typespec, l_typespec, u_typespec])

    call_target_name =
      case op_type do
        {:f, 32} ->
          "lu_cpu_custom_call_f32"

        {:f, 64} ->
          "lu_cpu_custom_call_f64"

        {:f, 16} ->
          "lu_cpu_custom_call_f16"

        {:bf, 16} ->
          "lu_cpu_custom_call_bf16"

        type ->
          # Due to matching on EXLA.Defn, we are sure that the device here is always :host
          raise "LU decomposition not supported on :host device for type #{inspect(type)}"
      end

    attributes = [
      call_target_name: attr_string(call_target_name),
      api_version: attr_i32(4)
    ]

    [p, l, u] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    # Convert p to the requested type if necessary
    p =
      if u8_typespec != p_typespec do
        convert(p, p_typespec)
      else
        p
      end

    {p, l, u}
  end

  # ===========================================================================
  # GPU Custom Calls (CUDA)
  # ===========================================================================

  @doc """
  GPU element-wise add via CUDA custom call.

  This is a prototype to validate GPU custom calls work in EXLA.
  The CUDA kernel is defined in `c_src/exla/custom_calls/gpu_add.cu`.

  ## Arguments

  - `a` - First tensor (f32)
  - `b` - Second tensor (f32, same shape as `a`)
  - `out_typespec` - Typespec for the output

  ## Returns

  Output value with element-wise sum.

  ## Note

  This only works when running on a CUDA device. The custom call target
  "exla_gpu_add_f32" is registered with platform "CUDA" in the C++ code.
  """
  def gpu_add(%Value{function: func} = a, %Value{function: func} = b, out_typespec) do
    %{type: a_type} = get_typespec(a)
    %{type: b_type} = get_typespec(b)

    unless a_type == {:f, 32} and b_type == {:f, 32} do
      raise ArgumentError,
            "gpu_add requires f32 tensors, got #{inspect(a_type)} and #{inspect(b_type)}"
    end

    operands = [a, b]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_gpu_add_f32"),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  # Fused MinGRU scan via CUDA custom call.
  #
  # Runs the entire sequential scan in a single CUDA kernel, avoiding per-timestep
  # graph breaks. The kernel is defined in `c_src/exla/custom_calls/fused_mingru_scan.cu`.
  #
  # Arguments:
  #   * `gates` - [batch, seq_len, hidden] f32 tensor (post-sigmoid gate values)
  #   * `candidates` - [batch, seq_len, hidden] f32 tensor
  #   * `h0` - [batch, hidden] f32 tensor (initial hidden state)
  #   * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, hidden})`

  # Derive precision suffix from a typespec for fused kernel dispatch.
  # bf16 tensors dispatch to bf16 kernels; everything else to f32.
  defp fused_precision_suffix(%{type: {:bf, 16}}), do: "bf16"
  defp fused_precision_suffix(_), do: "f32"

  def fused_mingru_scan(
        %Value{function: func} = gates,
        %Value{function: func} = candidates,
        %Value{function: func} = h0,
        out_typespec
      ) do
    operands = [gates, candidates, h0]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_mingru_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused MinLSTM scan via CUDA custom call.

  Runs the entire sequential scan in a single CUDA kernel. The kernel normalizes
  forget/input gates internally (f' = f/(f+i+eps), i' = i/(f+i+eps)).

  ## Arguments

    * `forget_gates` - [batch, seq_len, hidden] f32 tensor (post-sigmoid)
    * `input_gates` - [batch, seq_len, hidden] f32 tensor (post-sigmoid)
    * `candidates` - [batch, seq_len, hidden] f32 tensor
    * `h0` - [batch, hidden] f32 tensor (initial cell state)
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, hidden})`
  """
  def fused_minlstm_scan(
        %Value{function: func} = forget_gates,
        %Value{function: func} = input_gates,
        %Value{function: func} = candidates,
        %Value{function: func} = h0,
        out_typespec
      ) do
    operands = [forget_gates, input_gates, candidates, h0]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_minlstm_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused MinGRU multi-layer block scan via CUDA custom call.

  Processes all layers in a single kernel launch: LayerNorm → Dense projections →
  Sigmoid → Scan → Residual, with activations kept in shared memory/registers.

  ## Arguments

    * `input` - [batch, seq_len, hidden] f32 tensor
    * `weights` - [num_layers * (2*H*H + 4*H)] f32 flat packed weights
    * `h0` - [batch, num_layers, hidden] f32 tensor (per-layer initial states)
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, hidden})`
  """
  def fused_mingru_block_scan(
        %Value{function: func} = input,
        %Value{function: func} = weights,
        %Value{function: func} = h0,
        out_typespec
      ) do
    operands = [input, weights, h0]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_mingru_block_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused MinLSTM multi-layer block scan via CUDA custom call.

  Same as `fused_mingru_block_scan/4` but for MinLSTM with 3 projections
  and softmax-normalized forget/input gates.

  ## Arguments

    * `input` - [batch, seq_len, hidden] f32 tensor
    * `weights` - [num_layers * (3*H*H + 5*H)] f32 flat packed weights
    * `h0` - [batch, num_layers, hidden] f32 tensor (per-layer initial states)
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, hidden})`
  """
  def fused_minlstm_block_scan(
        %Value{function: func} = input,
        %Value{function: func} = weights,
        %Value{function: func} = h0,
        out_typespec
      ) do
    operands = [input, weights, h0]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_minlstm_block_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused linear multi-layer block scan via CUDA custom call.

  Processes all layers of a linear scan stack (h = a*h + b) in a single kernel.

  ## Arguments

    * `input` - [batch, seq_len, hidden] tensor
    * `weights` - [num_layers * (2*H*H + 4*H)] flat packed weights
    * `h0` - [batch, num_layers, hidden] per-layer initial states
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, hidden})`
  """
  def fused_linear_block_scan(
        %Value{function: func} = input,
        %Value{function: func} = weights,
        %Value{function: func} = h0,
        out_typespec
      ) do
    operands = [input, weights, h0]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_linear_block_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused LSTM multi-layer block scan via CUDA custom call.

  ## Arguments

    * `input` - [batch, seq_len, hidden] tensor
    * `weights` - [num_layers * (8*H*H + 6*H)] flat packed weights
    * `h0` - [batch, num_layers, hidden] per-layer initial hidden states
    * `c0` - [batch, num_layers, hidden] per-layer initial cell states
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, hidden})`
  """
  def fused_lstm_block_scan(
        %Value{function: func} = input,
        %Value{function: func} = weights,
        %Value{function: func} = h0,
        %Value{function: func} = c0,
        out_typespec
      ) do
    operands = [input, weights, h0, c0]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_lstm_block_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused GRU multi-layer block scan via CUDA custom call.

  ## Arguments

    * `input` - [batch, seq_len, hidden] tensor
    * `weights` - [num_layers * (6*H*H + 5*H)] flat packed weights
    * `h0` - [batch, num_layers, hidden] per-layer initial states
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, hidden})`
  """
  def fused_gru_block_scan(
        %Value{function: func} = input,
        %Value{function: func} = weights,
        %Value{function: func} = h0,
        out_typespec
      ) do
    operands = [input, weights, h0]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_gru_block_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused Liquid (LTC) exact solver scan via CUDA custom call.

  Implements x(t+1) = activation + (x(t) - activation) * exp(-1/tau) with dt=1.

  ## Arguments

    * `tau` - [batch, seq_len, hidden] f32 tensor (post-softplus time constants)
    * `activation` - [batch, seq_len, hidden] f32 tensor
    * `h0` - [batch, hidden] f32 tensor (initial hidden state)
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, hidden})`
  """
  def fused_liquid_scan(
        %Value{function: func} = tau,
        %Value{function: func} = activation,
        %Value{function: func} = h0,
        out_typespec
      ) do
    operands = [tau, activation, h0]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_liquid_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused ELU-GRU scan via CUDA custom call (NativeRecurrence variant).

  z = sigmoid(gate), c = 1 + elu(cand), h = (1-z)*h + z*c.
  Inputs are raw pre-activation projections — sigmoid/elu applied in-kernel.

  ## Arguments

    * `gates` - [batch, seq_len, hidden] f32 tensor (raw, pre-sigmoid)
    * `candidates` - [batch, seq_len, hidden] f32 tensor (raw, pre-elu)
    * `h0` - [batch, hidden] f32 tensor
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, hidden})`
  """
  def fused_elu_gru_scan(
        %Value{function: func} = gates,
        %Value{function: func} = candidates,
        %Value{function: func} = h0,
        out_typespec
      ) do
    operands = [gates, candidates, h0]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_elu_gru_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused Real-GRU scan via CUDA custom call (NativeRecurrence variant).

  z = sigmoid(gate), h = (1-z)*h + z*cand. Like MinGRU but with in-kernel sigmoid.

  ## Arguments

    * `gates` - [batch, seq_len, hidden] f32 tensor (raw, pre-sigmoid)
    * `candidates` - [batch, seq_len, hidden] f32 tensor
    * `h0` - [batch, hidden] f32 tensor
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, hidden})`
  """
  def fused_real_gru_scan(
        %Value{function: func} = gates,
        %Value{function: func} = candidates,
        %Value{function: func} = h0,
        out_typespec
      ) do
    operands = [gates, candidates, h0]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_real_gru_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused diagonal linear recurrence scan via CUDA custom call (NativeRecurrence variant).

  h = sigmoid(a)*h + b. The sigmoid is applied in-kernel.

  ## Arguments

    * `a_vals` - [batch, seq_len, hidden] f32 tensor (raw, pre-sigmoid)
    * `b_vals` - [batch, seq_len, hidden] f32 tensor
    * `h0` - [batch, hidden] f32 tensor
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, hidden})`
  """
  def fused_diag_linear_scan(
        %Value{function: func} = a_vals,
        %Value{function: func} = b_vals,
        %Value{function: func} = h0,
        out_typespec
      ) do
    operands = [a_vals, b_vals, h0]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_diag_linear_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused linear recurrence scan via CUDA custom call.

  h = a*h + b. No nonlinearities — all activations pre-computed on XLA side.
  Covers Griffin RG-LRU, MEGA EMA, GSS SSM, and other linear recurrences.

  ## Arguments

    * `a_vals` - [batch, seq_len, hidden] f32 tensor (multiplicative coefficients)
    * `b_vals` - [batch, seq_len, hidden] f32 tensor (additive terms)
    * `h0` - [batch, hidden] f32 tensor
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, hidden})`
  """
  def fused_linear_scan(
        %Value{function: func} = a_vals,
        %Value{function: func} = b_vals,
        %Value{function: func} = h0,
        out_typespec
      ) do
    operands = [a_vals, b_vals, h0]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_linear_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused linear scan backward via CUDA custom call.

  Multi-output: returns {grad_a, grad_b, grad_h0}.
  """
  def fused_linear_scan_backward(
        %Value{function: func} = a_vals,
        %Value{function: func} = h0,
        %Value{function: func} = forward_out,
        %Value{function: func} = grad_output,
        grad_a_typespec, grad_b_typespec, grad_h0_typespec
      ) do
    operands = [a_vals, h0, forward_out, grad_output]
    result_types = typespecs_to_mlir_types([grad_a_typespec, grad_b_typespec, grad_h0_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_linear_scan_backward_" <> fused_precision_suffix(grad_a_typespec)),
      api_version: attr_i32(4)
    ]

    [grad_a, grad_b, grad_h0] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {grad_a, grad_b, grad_h0}
  end

  @doc """
  Fused MinGRU scan backward via CUDA custom call.

  Multi-output: returns {grad_z, grad_cand, grad_h0}.
  """
  def fused_mingru_scan_backward(
        %Value{function: func} = z,
        %Value{function: func} = candidates,
        %Value{function: func} = h0,
        %Value{function: func} = forward_out,
        %Value{function: func} = grad_output,
        grad_z_typespec, grad_cand_typespec, grad_h0_typespec
      ) do
    operands = [z, candidates, h0, forward_out, grad_output]
    result_types = typespecs_to_mlir_types([grad_z_typespec, grad_cand_typespec, grad_h0_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_mingru_scan_backward_" <> fused_precision_suffix(grad_z_typespec)),
      api_version: attr_i32(4)
    ]

    [grad_z, grad_cand, grad_h0] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {grad_z, grad_cand, grad_h0}
  end

  @doc """
  Fused MinLSTM scan backward via CUDA custom call.

  Multi-output: returns {grad_f, grad_i, grad_cand, grad_h0}.
  """
  def fused_minlstm_scan_backward(
        %Value{function: func} = f,
        %Value{function: func} = i_gate,
        %Value{function: func} = candidates,
        %Value{function: func} = h0,
        %Value{function: func} = forward_out,
        %Value{function: func} = grad_output,
        grad_f_typespec, grad_i_typespec, grad_cand_typespec, grad_h0_typespec
      ) do
    operands = [f, i_gate, candidates, h0, forward_out, grad_output]
    result_types = typespecs_to_mlir_types([grad_f_typespec, grad_i_typespec, grad_cand_typespec, grad_h0_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_minlstm_scan_backward_" <> fused_precision_suffix(grad_f_typespec)),
      api_version: attr_i32(4)
    ]

    [grad_f, grad_i, grad_cand, grad_h0] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {grad_f, grad_i, grad_cand, grad_h0}
  end

  @doc """
  Fused ELU-GRU scan backward via CUDA custom call.

  Multi-output: returns {grad_z, grad_c, grad_h0}.
  """
  def fused_elu_gru_scan_backward(
        %Value{function: func} = z,
        %Value{function: func} = c,
        %Value{function: func} = h0,
        %Value{function: func} = forward_out,
        %Value{function: func} = grad_output,
        grad_z_typespec, grad_c_typespec, grad_h0_typespec
      ) do
    operands = [z, c, h0, forward_out, grad_output]
    result_types = typespecs_to_mlir_types([grad_z_typespec, grad_c_typespec, grad_h0_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_elu_gru_scan_backward_" <> fused_precision_suffix(grad_z_typespec)),
      api_version: attr_i32(4)
    ]

    [grad_z, grad_c, grad_h0] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {grad_z, grad_c, grad_h0}
  end

  @doc """
  Fused Real-GRU scan backward via CUDA custom call.

  Multi-output: returns {grad_z, grad_cand, grad_h0}.
  """
  def fused_real_gru_scan_backward(
        %Value{function: func} = z,
        %Value{function: func} = candidates,
        %Value{function: func} = h0,
        %Value{function: func} = forward_out,
        %Value{function: func} = grad_output,
        grad_z_typespec, grad_cand_typespec, grad_h0_typespec
      ) do
    operands = [z, candidates, h0, forward_out, grad_output]
    result_types = typespecs_to_mlir_types([grad_z_typespec, grad_cand_typespec, grad_h0_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_real_gru_scan_backward_" <> fused_precision_suffix(grad_z_typespec)),
      api_version: attr_i32(4)
    ]

    [grad_z, grad_cand, grad_h0] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {grad_z, grad_cand, grad_h0}
  end

  @doc """
  Fused DiagLinear scan backward via CUDA custom call.

  Multi-output: returns {grad_a, grad_b, grad_h0}.
  """
  def fused_diag_linear_scan_backward(
        %Value{function: func} = a_sig,
        %Value{function: func} = h0,
        %Value{function: func} = forward_out,
        %Value{function: func} = grad_output,
        grad_a_typespec, grad_b_typespec, grad_h0_typespec
      ) do
    operands = [a_sig, h0, forward_out, grad_output]
    result_types = typespecs_to_mlir_types([grad_a_typespec, grad_b_typespec, grad_h0_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_diag_linear_scan_backward_" <> fused_precision_suffix(grad_a_typespec)),
      api_version: attr_i32(4)
    ]

    [grad_a, grad_b, grad_h0] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {grad_a, grad_b, grad_h0}
  end

  @doc """
  Fused LSTM scan backward via CUDA custom call.

  Multi-output: returns {grad_wx, grad_h0, grad_c0}.
  """
  def fused_lstm_scan_backward(
        %Value{function: func} = wx,
        %Value{function: func} = r,
        %Value{function: func} = h0,
        %Value{function: func} = c0,
        %Value{function: func} = forward_out,
        %Value{function: func} = grad_output,
        grad_wx_typespec, grad_h0_typespec, grad_c0_typespec
      ) do
    operands = [wx, r, h0, c0, forward_out, grad_output]
    result_types = typespecs_to_mlir_types([grad_wx_typespec, grad_h0_typespec, grad_c0_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_lstm_scan_backward_" <> fused_precision_suffix(grad_wx_typespec)),
      api_version: attr_i32(4)
    ]

    [grad_wx, grad_h0, grad_c0] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {grad_wx, grad_h0, grad_c0}
  end

  @doc """
  Fused GRU scan backward via CUDA custom call.

  Multi-output: returns {grad_wx, grad_rh, grad_h0}.
  """
  def fused_gru_scan_backward(
        %Value{function: func} = wx,
        %Value{function: func} = r,
        %Value{function: func} = h0,
        %Value{function: func} = forward_out,
        %Value{function: func} = grad_output,
        grad_wx_typespec, grad_rh_typespec, grad_h0_typespec
      ) do
    operands = [wx, r, h0, forward_out, grad_output]
    result_types = typespecs_to_mlir_types([grad_wx_typespec, grad_rh_typespec, grad_h0_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_gru_scan_backward_" <> fused_precision_suffix(grad_wx_typespec)),
      api_version: attr_i32(4)
    ]

    [grad_wx, grad_rh, grad_h0] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {grad_wx, grad_rh, grad_h0}
  end

  @doc """
  Fused Selective Scan (Mamba) backward via CUDA custom call.

  Multi-output: returns {grad_x, grad_dt, grad_B, grad_C}.
  """
  def fused_selective_scan_backward(
        %Value{function: func} = x,
        %Value{function: func} = dt,
        %Value{function: func} = a,
        %Value{function: func} = b,
        %Value{function: func} = c,
        %Value{function: func} = grad_output,
        grad_x_typespec, grad_dt_typespec, grad_b_typespec, grad_c_typespec
      ) do
    operands = [x, dt, a, b, c, grad_output]
    result_types = typespecs_to_mlir_types([grad_x_typespec, grad_dt_typespec, grad_b_typespec, grad_c_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_selective_scan_backward_" <> fused_precision_suffix(grad_x_typespec)),
      api_version: attr_i32(4)
    ]

    [grad_x, grad_dt, grad_b, grad_c] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {grad_x, grad_dt, grad_b, grad_c}
  end

  @doc """
  Fused Liquid scan backward via CUDA custom call.

  Multi-output: returns {grad_tau, grad_act, grad_h0}.
  """
  def fused_liquid_scan_backward(
        %Value{function: func} = tau,
        %Value{function: func} = activation,
        %Value{function: func} = h0,
        %Value{function: func} = forward_out,
        %Value{function: func} = grad_output,
        grad_tau_typespec, grad_act_typespec, grad_h0_typespec
      ) do
    operands = [tau, activation, h0, forward_out, grad_output]
    result_types = typespecs_to_mlir_types([grad_tau_typespec, grad_act_typespec, grad_h0_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_liquid_scan_backward_" <> fused_precision_suffix(grad_tau_typespec)),
      api_version: attr_i32(4)
    ]

    [grad_tau, grad_act, grad_h0] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {grad_tau, grad_act, grad_h0}
  end

  @doc """
  Fused DeltaNet (delta rule) scan backward via CUDA custom call.

  Multi-output: returns {grad_q, grad_k, grad_v, grad_beta}.
  """
  def fused_delta_rule_scan_backward(
        %Value{function: func} = q,
        %Value{function: func} = k,
        %Value{function: func} = v,
        %Value{function: func} = beta,
        %Value{function: func} = forward_out,
        %Value{function: func} = grad_output,
        grad_q_typespec, grad_k_typespec, grad_v_typespec, grad_beta_typespec
      ) do
    operands = [q, k, v, beta, forward_out, grad_output]
    result_types = typespecs_to_mlir_types([grad_q_typespec, grad_k_typespec, grad_v_typespec, grad_beta_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_delta_rule_scan_backward_" <> fused_precision_suffix(grad_q_typespec)),
      api_version: attr_i32(4)
    ]

    [grad_q, grad_k, grad_v, grad_beta] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {grad_q, grad_k, grad_v, grad_beta}
  end

  @doc """
  Fused GatedDeltaNet scan backward via CUDA custom call.

  Multi-output: returns {grad_q, grad_k, grad_v, grad_beta, grad_alpha}.
  """
  def fused_gated_delta_net_scan_backward(
        %Value{function: func} = q,
        %Value{function: func} = k,
        %Value{function: func} = v,
        %Value{function: func} = beta,
        %Value{function: func} = alpha,
        %Value{function: func} = forward_out,
        %Value{function: func} = grad_output,
        grad_q_typespec, grad_k_typespec, grad_v_typespec, grad_beta_typespec, grad_alpha_typespec
      ) do
    operands = [q, k, v, beta, alpha, forward_out, grad_output]
    result_types = typespecs_to_mlir_types([grad_q_typespec, grad_k_typespec, grad_v_typespec, grad_beta_typespec, grad_alpha_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_gated_delta_net_scan_backward_" <> fused_precision_suffix(grad_q_typespec)),
      api_version: attr_i32(4)
    ]

    [grad_q, grad_k, grad_v, grad_beta, grad_alpha] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {grad_q, grad_k, grad_v, grad_beta, grad_alpha}
  end

  @doc """
  Fused DeltaProduct scan backward via CUDA custom call.

  Multi-output: returns {grad_q, grad_k, grad_v, grad_beta}.
  """
  def fused_delta_product_scan_backward(
        %Value{function: func} = q,
        %Value{function: func} = k,
        %Value{function: func} = v,
        %Value{function: func} = beta,
        %Value{function: func} = grad_output,
        grad_q_typespec, grad_k_typespec, grad_v_typespec, grad_beta_typespec
      ) do
    operands = [q, k, v, beta, grad_output]
    result_types = typespecs_to_mlir_types([grad_q_typespec, grad_k_typespec, grad_v_typespec, grad_beta_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_delta_product_scan_backward_" <> fused_precision_suffix(grad_q_typespec)),
      api_version: attr_i32(4)
    ]

    [grad_q, grad_k, grad_v, grad_beta] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {grad_q, grad_k, grad_v, grad_beta}
  end

  @doc """
  Fused sLSTM scan backward via CUDA custom call.

  Multi-output: returns {grad_wx, grad_h0, grad_c0}.
  """
  def fused_slstm_scan_backward(
        %Value{function: func} = wx,
        %Value{function: func} = r,
        %Value{function: func} = h0,
        %Value{function: func} = c0,
        %Value{function: func} = forward_out,
        %Value{function: func} = grad_output,
        grad_wx_typespec, grad_h0_typespec, grad_c0_typespec
      ) do
    operands = [wx, r, h0, c0, forward_out, grad_output]
    result_types = typespecs_to_mlir_types([grad_wx_typespec, grad_h0_typespec, grad_c0_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_slstm_scan_backward_" <> fused_precision_suffix(grad_wx_typespec)),
      api_version: attr_i32(4)
    ]

    [grad_wx, grad_h0, grad_c0] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {grad_wx, grad_h0, grad_c0}
  end

  @doc """
  Fused KDA (Kimi Delta Attention) scan backward via CUDA custom call.

  Multi-output: returns {grad_q, grad_k, grad_v, grad_alpha, grad_beta}.
  """
  def fused_kda_scan_backward(
        %Value{function: func} = q,
        %Value{function: func} = k,
        %Value{function: func} = v,
        %Value{function: func} = alpha,
        %Value{function: func} = beta,
        %Value{function: func} = forward_out,
        %Value{function: func} = grad_output,
        grad_q_typespec, grad_k_typespec, grad_v_typespec, grad_alpha_typespec, grad_beta_typespec
      ) do
    operands = [q, k, v, alpha, beta, forward_out, grad_output]
    result_types = typespecs_to_mlir_types([grad_q_typespec, grad_k_typespec, grad_v_typespec, grad_alpha_typespec, grad_beta_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_kda_scan_backward_" <> fused_precision_suffix(grad_q_typespec)),
      api_version: attr_i32(4)
    ]

    [grad_q, grad_k, grad_v, grad_alpha, grad_beta] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {grad_q, grad_k, grad_v, grad_alpha, grad_beta}
  end

  @doc """
  Fused RLA/RDN (Residual Linear Attention) scan backward via CUDA custom call.

  Multi-output: returns {grad_q, grad_k, grad_v, grad_alpha, grad_beta, grad_gamma}.
  """
  def fused_rla_scan_backward(
        %Value{function: func} = q,
        %Value{function: func} = k,
        %Value{function: func} = v,
        %Value{function: func} = alpha,
        %Value{function: func} = beta,
        %Value{function: func} = gamma,
        %Value{function: func} = fwd_grad_packed,
        grad_q_typespec, grad_k_typespec, grad_v_typespec,
        grad_alpha_typespec, grad_beta_typespec, grad_gamma_typespec
      ) do
    # 7 operands: forward_out and grad_output packed into [2,B,T,H,d] on MLIR side
    # to stay within XLA's 7-operand custom_call limit. Handler unpacks.
    # variant/clip hardcoded in handler (variant=0, clip=1.0).
    operands = [q, k, v, alpha, beta, gamma, fwd_grad_packed]
    result_types = typespecs_to_mlir_types([grad_q_typespec, grad_k_typespec, grad_v_typespec,
                                            grad_alpha_typespec, grad_beta_typespec, grad_gamma_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_rla_scan_backward_" <> fused_precision_suffix(grad_q_typespec)),
      api_version: attr_i32(4)
    ]

    [grad_q, grad_k, grad_v, grad_alpha, grad_beta, grad_gamma] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {grad_q, grad_k, grad_v, grad_alpha, grad_beta, grad_gamma}
  end

  @doc """
  Fused TTT-Linear scan backward via CUDA custom call.

  Multi-output: returns {grad_q, grad_k, grad_v, grad_eta, grad_w0, grad_lng, grad_lnb}.
  """
  def fused_ttt_scan_backward(
        %Value{function: func} = q,
        %Value{function: func} = k,
        %Value{function: func} = v,
        %Value{function: func} = eta,
        %Value{function: func} = w0,
        %Value{function: func} = ln_g,
        %Value{function: func} = ln_b,
        %Value{function: func} = forward_out,
        %Value{function: func} = grad_output,
        grad_q_typespec, grad_k_typespec, grad_v_typespec, grad_eta_typespec,
        grad_w0_typespec, grad_lng_typespec, grad_lnb_typespec
      ) do
    operands = [q, k, v, eta, w0, ln_g, ln_b, forward_out, grad_output]
    result_types = typespecs_to_mlir_types([grad_q_typespec, grad_k_typespec, grad_v_typespec,
                                            grad_eta_typespec, grad_w0_typespec, grad_lng_typespec, grad_lnb_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_ttt_scan_backward_" <> fused_precision_suffix(grad_q_typespec)),
      api_version: attr_i32(4)
    ]

    [grad_q, grad_k, grad_v, grad_eta, grad_w0, grad_lng, grad_lnb] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {grad_q, grad_k, grad_v, grad_eta, grad_w0, grad_lng, grad_lnb}
  end

  @doc """
  Fused DeltaNet scan via CUDA custom call.

  Matrix-state recurrence: S_t = S_{t-1} + beta * outer(v - S@k, k), o = S@q.
  Uses shared memory for the d x d state matrix.

  ## Arguments

    * `q` - [batch, seq_len, num_heads, head_dim] f32 tensor
    * `k` - [batch, seq_len, num_heads, head_dim] f32 tensor (L2-normalized)
    * `v` - [batch, seq_len, num_heads, head_dim] f32 tensor
    * `beta` - [batch, seq_len, num_heads, head_dim] f32 tensor (post-sigmoid)
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, num_heads, head_dim})`
  """
  def fused_delta_net_scan(
        %Value{function: func} = q,
        %Value{function: func} = k,
        %Value{function: func} = v,
        %Value{function: func} = beta,
        out_typespec
      ) do
    operands = [q, k, v, beta]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_delta_net_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused GatedDeltaNet scan via CUDA custom call.

  Like DeltaNet but with per-head scalar decay: S = alpha * S before delta update.

  ## Arguments

    * `q` - [batch, seq_len, num_heads, head_dim] f32 tensor
    * `k` - [batch, seq_len, num_heads, head_dim] f32 tensor (L2-normalized)
    * `v` - [batch, seq_len, num_heads, head_dim] f32 tensor
    * `beta` - [batch, seq_len, num_heads, head_dim] f32 tensor (post-sigmoid)
    * `alpha` - [batch, seq_len, num_heads] f32 tensor (per-head decay)
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, num_heads, head_dim})`
  """
  def fused_gated_delta_net_scan(
        %Value{function: func} = q,
        %Value{function: func} = k,
        %Value{function: func} = v,
        %Value{function: func} = beta,
        %Value{function: func} = alpha,
        out_typespec
      ) do
    operands = [q, k, v, beta, alpha]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_gated_delta_net_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused Mamba selective scan via CUDA custom call.

  Performs the full SSM recurrence with input-dependent discretization:
  A_bar = exp(dt * A), h_t = A_bar * h_{t-1} + dt*B*x_t, y_t = C*h_t.

  ## Arguments

    * `x` - [batch, seq_len, hidden] f32 tensor (input activations)
    * `dt` - [batch, seq_len, hidden] f32 tensor (discretization timesteps)
    * `a` - [hidden, state] f32 tensor (state transition diagonal, negative)
    * `b` - [batch, seq_len, state] f32 tensor (input-to-state projection)
    * `c` - [batch, seq_len, state] f32 tensor (state-to-output projection)
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, hidden})`
  """
  def fused_selective_scan(
        %Value{function: func} = x,
        %Value{function: func} = dt,
        %Value{function: func} = a,
        %Value{function: func} = b,
        %Value{function: func} = c,
        out_typespec
      ) do
    operands = [x, dt, a, b, c]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_selective_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused DeltaProduct scan via CUDA custom call.

  Multiple Householder transformation steps per token with state matrix S:
  S = (I - beta*k*k^T) @ S + beta*k*v^T, output = RMS_norm(S @ q).
  Keys are L2-normalized in-kernel.

  ## Arguments

    * `q` - [batch, seq_len, num_heads, head_dim] f32 tensor
    * `k` - [batch, seq_len, num_householder, num_heads, head_dim] f32 tensor
    * `v` - [batch, seq_len, num_householder, num_heads, head_dim] f32 tensor
    * `beta` - [batch, seq_len, num_householder, num_heads] f32 tensor (post-sigmoid)
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, num_heads, head_dim})`
  """
  def fused_delta_product_scan(
        %Value{function: func} = q,
        %Value{function: func} = k,
        %Value{function: func} = v,
        %Value{function: func} = beta,
        out_typespec
      ) do
    operands = [q, k, v, beta]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_delta_product_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused sLSTM scan via CUDA custom call.

  sLSTM (scalar LSTM with exponential gating) uses log-domain stabilized gates:
  m_t = max(log_f + m_{t-1}, log_i), i/f = exp(log_gate - m_t),
  c = f*c + i*z, n = f*n + i, h = o * c/max(|n|, 1).

  ## Arguments

    * `wx` - [batch, seq_len, 4*hidden] f32 tensor (pre-computed W@x for i,f,z,o gates)
    * `r` - [hidden, 4*hidden] f32 tensor (recurrent weight matrix)
    * `h0` - [batch, hidden] f32 tensor (initial hidden state)
    * `c0` - [batch, hidden] f32 tensor (initial cell state)
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, hidden})`
  """
  def fused_slstm_scan(
        %Value{function: func} = wx,
        %Value{function: func} = r,
        %Value{function: func} = h0,
        %Value{function: func} = c0,
        out_typespec
      ) do
    operands = [wx, r, h0, c0]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_slstm_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused LSTM scan via CUDA custom call.

  Standard LSTM with hidden-to-hidden matmul R@h fused into the scan:
  i/f/o = sigmoid(wx + rh), g = tanh(wx + rh), c = f*c + i*g, h = o*tanh(c).

  ## Arguments

    * `wx` - [batch, seq_len, 4*hidden] f32 tensor (pre-computed W@x + bias)
    * `r` - [hidden, 4*hidden] f32 tensor (recurrent weight matrix)
    * `h0` - [batch, hidden] f32 tensor (initial hidden state)
    * `c0` - [batch, hidden] f32 tensor (initial cell state)
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, hidden})`
  """
  def fused_lstm_scan(
        %Value{function: func} = wx,
        %Value{function: func} = r,
        %Value{function: func} = h0,
        %Value{function: func} = c0,
        out_typespec
      ) do
    operands = [wx, r, h0, c0]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_lstm_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused GRU scan via CUDA custom call.

  Standard GRU with hidden-to-hidden matmul R@h fused into the scan:
  z = sigmoid(wx_z + rh_z), r = sigmoid(wx_r + rh_r),
  h_tilde = tanh(wx_h + r * rh_h), h = (1-z)*h_tilde + z*h_prev.

  ## Arguments

    * `wx` - [batch, seq_len, 3*hidden] f32 tensor (pre-computed W@x + bias)
    * `r` - [hidden, 3*hidden] f32 tensor (recurrent weight matrix)
    * `h0` - [batch, hidden] f32 tensor (initial hidden state)
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, hidden})`
  """
  def fused_gru_scan(
        %Value{function: func} = wx,
        %Value{function: func} = r,
        %Value{function: func} = h0,
        out_typespec
      ) do
    operands = [wx, r, h0]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_gru_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused TTT-Linear (Test-Time Training) scan via CUDA custom call.

  TTT uses an inner linear model W as hidden state, updated per timestep:
  pred = W@k, error = LN(pred) - v, W -= eta * error @ k^T, out = W@q.

  ## Arguments

    * `q` - [batch, seq_len, inner_size] f32 tensor (query projections)
    * `k` - [batch, seq_len, inner_size] f32 tensor (key projections)
    * `v` - [batch, seq_len, inner_size] f32 tensor (value/target projections)
    * `eta` - [batch, seq_len, inner_size] f32 tensor (learning rate, post-sigmoid/scaled)
    * `w0` - [batch, inner_size, inner_size] f32 tensor (initial weight matrix)
    * `ln_g` - [inner_size] f32 tensor (LayerNorm gamma)
    * `ln_b` - [inner_size] f32 tensor (LayerNorm beta)
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, seq_len, inner_size})`
  """
  def fused_ttt_scan(
        %Value{function: func} = q,
        %Value{function: func} = k,
        %Value{function: func} = v,
        %Value{function: func} = eta,
        %Value{function: func} = w0,
        %Value{function: func} = ln_g,
        %Value{function: func} = ln_b,
        out_typespec
      ) do
    operands = [q, k, v, eta, w0, ln_g, ln_b]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_ttt_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  def fused_kda_scan(
        %Value{function: func} = q,
        %Value{function: func} = k,
        %Value{function: func} = v,
        %Value{function: func} = alpha,
        %Value{function: func} = beta,
        out_typespec
      ) do
    operands = [q, k, v, alpha, beta]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_kda_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  def fused_rla_scan(
        %Value{function: func} = q,
        %Value{function: func} = k,
        %Value{function: func} = v,
        %Value{function: func} = alpha,
        %Value{function: func} = beta,
        %Value{function: func} = gamma,
        out_typespec
      ) do
    # 6 operands only — variant/clip omitted due to XLA 7-operand limit on custom_calls.
    # Handler hardcodes variant=0, clip=1.0.
    operands = [q, k, v, alpha, beta, gamma]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_rla_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused Flash Attention V2 via CUDA custom call.

  IO-aware exact attention with tiled loading and online softmax.
  Avoids materializing the full [seq, seq] attention matrix.

  ## Arguments

    * `q` - [batch, heads, seq_len, head_dim] f32 tensor
    * `k` - [batch, heads, seq_len, head_dim] f32 tensor
    * `v` - [batch, heads, seq_len, head_dim] f32 tensor
    * `causal` - scalar i32 tensor (0 = full, 1 = causal mask)
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, heads, seq_len, head_dim})`
  """
  def fused_flash_attention(
        %Value{function: func} = q,
        %Value{function: func} = k,
        %Value{function: func} = v,
        out_typespec
      ) do
    operands = [q, k, v]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_flash_attention_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  LASER flash attention custom call.

  Computes `log(softmax(QK^T / sqrt(d)) @ exp(V))` using the LWSE trick.

  ## Operands
    * `q` - [batch, heads, seq_len, head_dim] f32 tensor
    * `k` - [batch, heads, seq_len, head_dim] f32 tensor
    * `v` - [batch, heads, seq_len, head_dim] f32 tensor
    * `v_max` - [batch, heads, 1, head_dim] f32 tensor
    * `causal` - scalar i32 tensor (0 = full, 1 = causal mask)
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, heads, seq_len, head_dim})`
  """
  def fused_laser_attention(
        %Value{function: func} = q,
        %Value{function: func} = k,
        %Value{function: func} = v,
        %Value{function: func} = v_max,
        out_typespec
      ) do
    operands = [q, k, v, v_max]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_laser_attention_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  FoX flash attention custom call.

  Computes `softmax(QK^T / sqrt(d) + forget_bias) @ V` with
  cumulative log-forget bias fused into the tiled sweep.

  ## Operands
    * `q` - [batch, heads, seq_len, head_dim] f32 tensor
    * `k` - [batch, heads, seq_len, head_dim] f32 tensor
    * `v` - [batch, heads, seq_len, head_dim] f32 tensor
    * `cs` - [batch, heads, seq_len] f32 tensor (cumulative log-forget)
    * `out_typespec` - `Typespec.tensor({:f, 32}, {batch, heads, seq_len, head_dim})`
  """
  def fused_fox_attention(
        %Value{function: func} = q,
        %Value{function: func} = k,
        %Value{function: func} = v,
        %Value{function: func} = cs,
        out_typespec
      ) do
    operands = [q, k, v, cs]
    result_types = typespecs_to_mlir_types([out_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_fox_attention_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Fused Flash Attention backward via CUDA custom call.

  Multi-output: returns {dQ, dK, dV}.
  """
  def fused_flash_attention_backward(
        %Value{function: func} = q,
        %Value{function: func} = k,
        %Value{function: func} = v,
        %Value{function: func} = o,
        %Value{function: func} = grad_o,
        dq_typespec, dk_typespec, dv_typespec
      ) do
    operands = [q, k, v, o, grad_o]
    result_types = typespecs_to_mlir_types([dq_typespec, dk_typespec, dv_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_flash_attention_backward_" <> fused_precision_suffix(dq_typespec)),
      api_version: attr_i32(4)
    ]

    [dq, dk, dv] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {dq, dk, dv}
  end

  @doc """
  Fused LASER Attention backward via CUDA custom call.

  Multi-output: returns {dQ, dK, dV}.
  """
  def fused_laser_attention_backward(
        %Value{function: func} = q,
        %Value{function: func} = k,
        %Value{function: func} = v,
        %Value{function: func} = v_max,
        %Value{function: func} = o,
        %Value{function: func} = grad_o,
        dq_typespec, dk_typespec, dv_typespec
      ) do
    operands = [q, k, v, v_max, o, grad_o]
    result_types = typespecs_to_mlir_types([dq_typespec, dk_typespec, dv_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_laser_attention_backward_" <> fused_precision_suffix(dq_typespec)),
      api_version: attr_i32(4)
    ]

    [dq, dk, dv] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {dq, dk, dv}
  end

  @doc """
  Fused FoX Attention backward via CUDA custom call.

  Multi-output: returns {dQ, dK, dV, grad_cs}.
  """
  def fused_fox_attention_backward(
        %Value{function: func} = q,
        %Value{function: func} = k,
        %Value{function: func} = v,
        %Value{function: func} = cs,
        %Value{function: func} = o,
        %Value{function: func} = grad_o,
        dq_typespec, dk_typespec, dv_typespec, dcs_typespec
      ) do
    operands = [q, k, v, cs, o, grad_o]
    result_types = typespecs_to_mlir_types([dq_typespec, dk_typespec, dv_typespec, dcs_typespec])

    attributes = [
      call_target_name: attr_string("exla_fused_fox_attention_backward_" <> fused_precision_suffix(dq_typespec)),
      api_version: attr_i32(4)
    ]

    [dq, dk, dv, dcs] =
      op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)

    {dq, dk, dv, dcs}
  end

  def fused_reservoir_scan(
        %Value{function: func} = wx,
        %Value{function: func} = w_res,
        %Value{function: func} = h0,
        out_typespec
      ) do
    operands = [wx, w_res, h0]
    result_types = typespecs_to_mlir_types([out_typespec])
    attributes = [
      call_target_name: attr_string("exla_fused_reservoir_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]
    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  def fused_titans_scan(
        %Value{function: func} = combined,
        out_typespec
      ) do
    operands = [combined]
    result_types = typespecs_to_mlir_types([out_typespec])
    attributes = [
      call_target_name: attr_string("exla_fused_titans_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]
    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  def fused_miras_scan(
        %Value{function: func} = combined,
        out_typespec
      ) do
    operands = [combined]
    result_types = typespecs_to_mlir_types([out_typespec])
    attributes = [
      call_target_name: attr_string("exla_fused_miras_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]
    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  def fused_gsa_scan(
        %Value{function: func} = q,
        %Value{function: func} = k_slot,
        %Value{function: func} = v,
        %Value{function: func} = alpha,
        out_typespec
      ) do
    operands = [q, k_slot, v, alpha]
    result_types = typespecs_to_mlir_types([out_typespec])
    attributes = [
      call_target_name: attr_string("exla_fused_gsa_scan_" <> fused_precision_suffix(out_typespec)),
      api_version: attr_i32(4)
    ]
    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes) |> one!()
  end

  @doc """
  Builds a StableHLO `custom_call` that targets the EXLA Elixir callback bridge.

  The `callback_id` is typically the underlying `Nx.Defn.Expr` id of the
  `:runtime_call` node. It is encoded as a binary (via `:erlang.term_to_binary/1`)
  and then represented as a list of 64-bit words in the custom call attributes,
  similar to how we encode the callback server PID.
  """
  def runtime_call(
        [%Value{function: func} | _] = operands,
        typespecs,
        callback_server_pid,
        callback_id
      ) do
    result_types = typespecs_to_mlir_types(typespecs)

    {callback_server_pid_words, callback_server_pid_size} =
      term_to_int64_list(callback_server_pid)

    {callback_id_words, callback_id_size} =
      term_to_int64_list(callback_id)

    attributes = [
      call_target_name: attr_string("exla_runtime_callback"),
      # api_version 4 enables the typed FFI API used by our callback handler.
      api_version: attr_i32(4),
      backend_config:
        attr_dict(
          callback_id: attr_array_i64_elements(callback_id_words),
          callback_id_size: attr_ui64(callback_id_size),
          callback_server_pid: attr_array_i64_elements(callback_server_pid_words),
          callback_server_pid_size: attr_ui64(callback_server_pid_size)
        )
    ]

    op(func, "stablehlo.custom_call", operands, result_types, attributes: attributes)
  end

  defp term_to_int64_list(term) do
    bin = :erlang.term_to_binary(term)
    size = byte_size(bin)

    # Zero-pad the binary so its size is a multiple of 8 and it can be
    # represented as a list of 64-bit words.
    pad =
      case rem(size, 8) do
        0 -> 0
        r -> 8 - r
      end

    padded_bin =
      if pad == 0 do
        bin
      else
        bin <> :binary.copy(<<0>>, pad)
      end

    words =
      for <<x::unsigned-native-64 <- padded_bin>> do
        x
      end

    {words, size}
  end

  def get_tuple_element(%Value{function: func} = operand, index, typespec) do
    result_types = typespecs_to_mlir_types([typespec])
    attributes = [index: attr_i32(index)]

    op(func, "stablehlo.get_tuple_element", [operand], result_types, attributes: attributes)
    |> one!()
  end

  def get_typespec(value) do
    EXLA.NIF.mlir_get_typespec(value.ref)
  end

  def typespecs_to_mlir_types(shapes) do
    Enum.map(shapes, &typespec_to_mlir_type/1)
  end

  defp typespec_to_mlir_type(%{type: :token}), do: type_token()
  defp typespec_to_mlir_type(%{type: type, shape: shape}), do: type_tensor(type, shape)

  defp one!([value]), do: value

  defp one!(other) do
    raise "expected a list with single element, got: #{inspect(other)}"
  end

  defp complex_part_type({:c, size}), do: {:f, div(size, 2)}

  defp op(%Function{} = function, op_name, operands, result_types, opts \\ []) do
    opts = Keyword.validate!(opts, attributes: [], regions: [])

    %{ref: function_ref} = function

    refs =
      Enum.map(operands, fn
        %Value{ref: ref, function: %Function{ref: ^function_ref}} -> ref
      end)

    refs =
      EXLA.NIF.mlir_op(
        function.ref,
        op_name,
        refs,
        result_types,
        opts[:attributes],
        opts[:regions]
      )

    Enum.map(refs, &%Value{function: function, ref: &1})
  end

  defp type_tensor(type, shape) do
    shape_sequence = shape |> Tuple.to_list() |> Enum.map_join("", &"#{&1}x")
    "tensor<#{shape_sequence}#{type_number(type)}>"
  end

  defp type_number({:pred, 8}), do: "i1"
  defp type_number({:s, width}), do: "i#{width}"
  defp type_number({:u, width}), do: "ui#{width}"
  defp type_number({:f, 8}), do: "f8E5M2"
  defp type_number({:f8_e4m3fn, 8}), do: "f8E4M3FN"
  defp type_number({:f, width}), do: "f#{width}"
  defp type_number({:bf, width}), do: "bf#{width}"
  defp type_number({:c, 64}), do: "complex<f32>"
  defp type_number({:c, 128}), do: "complex<f64>"

  defp type_token(), do: "!stablehlo.token"

  defp number_literal(value, type) do
    cond do
      Nx.Type.complex?(type) ->
        {re, im} =
          case value do
            %Complex{re: re, im: im} -> {re, im}
            true -> {1, 0}
            false -> {0, 0}
            n -> {n, 0}
          end

        subtype = complex_part_type(type)
        "(#{number_literal(re, subtype)}, #{number_literal(im, subtype)})"

      Nx.Type.float?(type) ->
        # We pass floats using binary representation, because that is
        # likely more robust and not a subject to formatting limits and
        # rounding. Based on the examples in the docs, the hexadecimal
        # representation is always big-endian.
        #
        # See https://mlir.llvm.org/docs/Dialects/Builtin/#floatattr
        hex_data = float_hex(value, type)
        "0x#{hex_data}"

      true ->
        "#{value}"
    end
  end

  defp float_hex(value, {mod, size} = type) do
    data =
      case value do
        :nan -> type |> Nx.Type.nan_binary() |> native_to_big()
        :infinity -> type |> Nx.Type.infinity_binary() |> native_to_big()
        :neg_infinity -> type |> Nx.Type.neg_infinity_binary() |> native_to_big()
        value when mod == :f8_e4m3fn -> Nx.Floating.dump_f8_e4m3fn(value)
        value when size == 8 -> Nx.Floating.dump_f8(value)
        value when mod == :bf and size == 16 -> bf16_to_big(value)
        value -> <<value::float-size(size)-big>>
      end

    Base.encode16(data)
  end

  defp bf16_to_big(x) do
    binary_part(<<x::float-big-32>>, 0, 2)
  end

  defp native_to_big(binary) do
    size = byte_size(binary) * 8
    <<value::size(^size)-native>> = binary
    <<value::size(size)-big>>
  end

  defp attr_array_i64_elements([]) do
    "array<i64>"
  end

  defp attr_array_i64_elements(list) do
    "array<i64: #{Enum.join(list, ", ")}>"
  end

  defp attr_dense_elements([], type, {0} = shape) do
    "dense<[]> : #{type_tensor(type, shape)}"
  end

  defp attr_dense_elements(list, type, shape) do
    literals = Enum.map(list, &number_literal(&1, type))

    list_literal =
      shape
      |> Tuple.to_list()
      |> List.foldr(literals, fn size, acc ->
        acc
        |> Enum.chunk_every(size)
        |> Enum.map(fn chunk ->
          ["[", Enum.intersperse(chunk, ", "), "]"]
        end)
      end)
      |> IO.iodata_to_binary()

    "dense<#{list_literal}> : #{type_tensor(type, shape)}"
  end

  defp attr_string(string), do: ~s["#{string}"]

  defp attr_symbol_reference(id), do: "@#{id}"

  defp attr_boolean(true), do: "true"
  defp attr_boolean(false), do: "false"

  defp attr_i32(number), do: "#{number} : i32"

  defp attr_i64(number), do: "#{number} : i64"
  defp attr_ui64(number), do: "#{number} : ui64"

  defp attr_padding(padding) do
    list = Enum.flat_map(padding, &Tuple.to_list/1)
    attr_dense_elements(list, {:s, 64}, {length(padding), 2})
  end

  defp attr_comparison_direction(value) when value in [:eq, :lt, :le, :gt, :ge, :ne],
    do: attr_enum("stablehlo", "comparison_direction", value)

  defp attr_comparison_type(value) when value in [:float, :totalorder],
    do: attr_enum("stablehlo", "comparison_type", value)

  defp attr_precision(value) when value in [:default, :high, :highest],
    do: attr_enum("stablehlo", "precision", value)

  defp attr_transpose(value) when value in [:adjoint, :transpose, :no_transpose],
    do: attr_enum("stablehlo", "transpose", value)

  defp attr_fft_type(value) when value in [:fft, :ifft],
    do: attr_enum("stablehlo", "fft_type", value)

  defp attr_enum(dialect, enum_name, value) do
    value = value |> Atom.to_string() |> String.upcase()
    "##{dialect}<#{enum_name} #{value}>"
  end

  defp attr_struct(name, keyword_list) do
    content = Enum.map_join(keyword_list, ", ", fn {key, value} -> "#{key} = #{value}" end)
    "##{name}<#{content}>"
  end

  defp attr_dict(keyword_list) do
    content = Enum.map_join(keyword_list, ", ", fn {key, value} -> "#{key} = #{value}" end)
    "{#{content}}"
  end

  defp join_list(list) do
    "[" <> Enum.join(list, ", ") <> "]"
  end
end
