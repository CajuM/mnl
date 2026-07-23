{ pkgs ? import (fetchTarball https://github.com/NixOS/nixpkgs/archive/4382ed2b7a6839d4280a9b386db49cbc5907414d.tar.gz) { config.rocmSupport = true; } }:

pkgs.mkShell {
  name = "mnl-shell";

  packages = [
    (pkgs.python3.withPackages (ps: with ps; [
      matplotlib
      optuna
      torch
      torchaudio
    ]))
  ];
}
