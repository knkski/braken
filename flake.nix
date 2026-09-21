{
  description = "Braken L-system tools and applications";

  inputs = {
    cuda-oxide.url = "github:NVlabs/cuda-oxide";
    nixpkgs.follows = "cuda-oxide/nixpkgs";
    nixpkgsUnstable.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.follows = "cuda-oxide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      cuda-oxide,
      nixpkgs,
      nixpkgsUnstable,
      flake-utils,
      rust-overlay,
      ...
    }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };
        pkgsUnstable = import nixpkgsUnstable {
          inherit system;
        };
        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        nvidiaDriverLib = "/run/opengl-driver/lib";
        guiRuntimeLibs = pkgs.lib.makeLibraryPath [
          pkgs.wayland
          pkgs.libxkbcommon
          pkgs.libxcb
          pkgs.libx11
          pkgs.vulkan-loader
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          inputsFrom = [ cuda-oxide.devShells.${system}.default ];
          packages = [
            rustToolchain
            pkgs.wayland
            pkgs.libxkbcommon
            pkgs.libxcb
            pkgs.libx11
            pkgs.vulkan-loader
            pkgs.trunk
            pkgs.just
            pkgsUnstable.codex
          ];

          shellHook = ''
            export PATH="${rustToolchain}/bin:''${PATH}"
            export LD_LIBRARY_PATH="${guiRuntimeLibs}:''${LD_LIBRARY_PATH:-}"

            if [ -e "${nvidiaDriverLib}/libcuda.so.1" ]; then
              export LD_LIBRARY_PATH="${nvidiaDriverLib}:''${LD_LIBRARY_PATH:-}"
              echo "CUDA driver: using ${nvidiaDriverLib}"
            else
              echo "CUDA driver: ${nvidiaDriverLib}/libcuda.so.1 not found"
              echo "CUDA driver: enter this shell on the NixOS host with the NVIDIA driver installed"
            fi

            if command -v nvidia-smi >/dev/null 2>&1; then
              nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
            fi
          '';
        };
      }
    );
}
