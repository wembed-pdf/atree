{
  description = "SPRK-tree - Spatial index data structure with C/C++ and Python bindings";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        toolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = ["rust-src" "clippy" "rust-analyzer"];
        };

        pkgs = import nixpkgs {
          inherit system overlays;
        };

        python-with-packages = pkgs.python3.withPackages (ps: with ps; [
          numpy
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            git
            python-with-packages
          ];

          packages = with pkgs; [
            toolchain
            maturin
            cargo-semver-checks
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
          ];
        };
      }
    );
}
