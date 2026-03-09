defmodule EXLA.ClientTest do
  use ExUnit.Case, async: true

  doctest EXLA.Client

  describe "get_supported_platforms/0" do
    test "returns supported platforms with device information" do
      %{host: _} = EXLA.Client.get_supported_platforms()
    end
  end

  describe "allocator option" do
    test "raises on invalid allocator for GPU client" do
      clients = Application.get_env(:exla, :clients)

      Application.put_env(
        :exla,
        :clients,
        clients ++ [bad_alloc: [platform: :cuda, allocator: :invalid]]
      )

      # Clear any cached client
      :persistent_term.erase({EXLA.Client, :bad_alloc})

      if Map.has_key?(EXLA.Client.get_supported_platforms(), :cuda) do
        {{%ArgumentError{message: message}, _stacktrace}, _call} =
          catch_exit(EXLA.Client.fetch!(:bad_alloc))

        assert message =~ "invalid :allocator option"
      end
    after
      :persistent_term.erase({EXLA.Client, :bad_alloc})
    end

    test "valid allocator options are accepted for GPU client" do
      if Map.has_key?(EXLA.Client.get_supported_platforms(), :cuda) do
        for allocator <- [:bfc, :cuda_async, :default] do
          client_name = :"test_alloc_#{allocator}"
          clients = Application.get_env(:exla, :clients)

          Application.put_env(
            :exla,
            :clients,
            clients ++ [{client_name, [platform: :cuda, allocator: allocator]}]
          )

          client = EXLA.Client.fetch!(client_name)
          assert client.platform == :cuda
          :persistent_term.erase({EXLA.Client, client_name})
        end
      end
    end
  end
end
