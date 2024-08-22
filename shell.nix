{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  packages = with pkgs; [
    (python312.withPackages (
      ps: with ps; [
        ipython
        matplotlib
        torch
        torchaudio
      ]
    ))
  ];
}
