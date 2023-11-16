{
  inputs.nixpkgs.url = "nixpkgs";
  inputs.utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, utils }:
    (utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShell = (pkgs.buildFHSUserEnv {
          name = "mnl";
          targetPkgs = ps: with ps; [
            python311Full
            ffmpeg.lib
            gcc.cc.lib
            zlib
          ];
        }).env;
      }));
}
