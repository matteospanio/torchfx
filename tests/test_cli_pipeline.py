"""Tests for the SoX-style pipeline parser and the ``compile`` (.fxg) artifact."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
from typer.testing import CliRunner

from cli.app import app
from cli.commands.compile import load_compiled, save_compiled
from cli.parsing import parse_pipeline, parse_pipeline_specs
from torchfx.filter import SOSFilter

runner = CliRunner()


class TestPipelineParser:
    def test_positional_form(self):
        chain = parse_pipeline("lobutterworth --cutoff 800 --order 4 gain --gain 0.5")
        names = [type(m).__name__ for m in chain.children()]
        assert names == ["LoButterworth", "Gain"]
        assert chain[0].cutoff == 800

    def test_pipe_delimited_form(self):
        chain = parse_pipeline("lobutterworth --cutoff 800 --order 4 | hibutterworth --cutoff 100")
        assert [type(m).__name__ for m in chain.children()] == ["LoButterworth", "HiButterworth"]

    def test_token_list_form(self):
        chain = parse_pipeline(["gain", "--gain", "0.5"])
        assert [type(m).__name__ for m in chain.children()] == ["Gain"]

    def test_flag_value_that_is_an_effect_name(self):
        # 'lowpass' here is the VALUE of --btype, not a new effect.
        specs = parse_pipeline_specs("butterworth --btype lowpass --cutoff 1000 --order 4")
        assert specs == [("butterworth", {"btype": "lowpass", "cutoff": 1000, "order": 4})]

    def test_unknown_effect_raises(self):
        import pytest

        with pytest.raises(ValueError, match="notareal"):
            parse_pipeline("notareal --x 1")


class TestCompiledArtifact:
    def test_roundtrip_filters_become_sos(self, tmp_path: Path):
        path = tmp_path / "chain.fxg"
        save_compiled(
            "lobutterworth --cutoff 800 --order 4 | hibutterworth --cutoff 100", 48000, path
        )
        chain = load_compiled(path)
        assert all(isinstance(m, SOSFilter) for m in chain.children())
        y = chain(torch.randn(1, 2000))
        assert y.shape == (1, 2000)

    def test_roundtrip_keeps_non_filter_effects(self, tmp_path: Path):
        path = tmp_path / "mixed.fxg"
        save_compiled("lobutterworth --cutoff 800 --order 4 | reverb --mix 0.3", 48000, path)
        names = [type(m).__name__ for m in load_compiled(path).children()]
        assert names == ["SOSFilter", "Reverb"]

    def test_compiled_matches_direct_pipeline(self, tmp_path: Path):
        path = tmp_path / "c.fxg"
        spec = "lobutterworth --cutoff 800 --order 4 | hibutterworth --cutoff 100 --order 2"
        save_compiled(spec, 48000, path)
        compiled = load_compiled(path)

        direct = parse_pipeline(spec).compile(48000)
        x = torch.randn(1, 4000)
        torch.testing.assert_close(compiled(x), direct(x), atol=1e-9, rtol=1e-6)


class TestCliCommands:
    def test_compile_then_process(self, tmp_path: Path):
        src = tmp_path / "in.wav"
        sf.write(str(src), (0.1 * torch.randn(4800)).numpy(), 48000)
        fxg = tmp_path / "chain.fxg"
        out = tmp_path / "out.wav"

        r1 = runner.invoke(
            app,
            ["compile", "lobutterworth --cutoff 800 --order 4", "--fs", "48000", "-o", str(fxg)],
        )
        assert r1.exit_code == 0, r1.output
        assert fxg.exists()

        r2 = runner.invoke(app, ["process", str(src), str(out), "--compiled", str(fxg)])
        assert r2.exit_code == 0, r2.output
        assert out.exists()

    def test_pipe_command(self, tmp_path: Path):
        src = tmp_path / "in.wav"
        sf.write(str(src), (0.1 * torch.randn(4800)).numpy(), 48000)
        out = tmp_path / "out.wav"
        r = runner.invoke(
            app, ["pipe", str(src), str(out), "lobutterworth", "--cutoff", "800", "--order", "4"]
        )
        assert r.exit_code == 0, r.output
        assert out.exists()
