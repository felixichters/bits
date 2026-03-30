{
  description = "RevEng ML - binary boundary detection dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          name = "reveng-ml";

          packages = with pkgs; [
            # Python + uv for managing the virtualenv
            python311
            uv

            # Required by data.py at runtime
            binutils       # provides `strip`

            # Required for compiling test binaries in tests/
            gcc

            # Useful for debugging ELF files
            bintools-unwrapped  # provides `objdump`, `readelf`, `nm`
          ];

          # uv needs to be able to link compiled Python extensions
          # Point it at Nix's glibc and zlib
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
            stdenv.cc.cc.lib   # libstdc++
            zlib
            libz
          ]);

          shellHook = ''
            echo "RevEng ML dev environment"

            echo "Run: uv sync && uv run python -m reveng_ml --help"
            echo "Note: uses CPU torch. On bwUniCluster run: uv sync --extra cuda"
          '';
        };
      });
}
