{
  description = "Bevy Outliner - Jump Flood Algorithm based object outlining for Bevy 0.18";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };
      in
      {
        devShells.default = pkgs.mkShell rec {
          nativeBuildInputs = with pkgs; [
            rustToolchain
            pkg-config
            mold
            clang
          ];

          buildInputs = with pkgs; [
            # Bevy dependencies
            udev
            alsa-lib
            vulkan-loader

            # X11
            xorg.libX11
            xorg.libXcursor
            xorg.libXi
            xorg.libXrandr

            # Wayland
            libxkbcommon
            wayland

            # Other
            libGL
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;

          RUST_BACKTRACE = 1;

          shellHook = ''
            echo "Bevy Outliner development environment"
            echo "Rust: $(rustc --version)"
          '';
        };
      }
    );
}
