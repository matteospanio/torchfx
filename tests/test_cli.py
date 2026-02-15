"""Tests for the TorchFX CLI application."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torchaudio
from typer.testing import CliRunner

from cli.app import app
from cli.parsing import (
    EFFECT_REGISTRY,
    _coerce_value,
    list_effects,
    load_effects_from_config,
    parse_effect_list,
    parse_effect_string,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def wav_file(tmp_path: Path) -> Path:
    """Create a short mono WAV file for testing."""
    path = tmp_path / "test.wav"
    waveform = torch.randn(1, 44100)  # 1 second mono
    torchaudio.save(str(path), waveform, 44100)
    return path


@pytest.fixture()
def stereo_wav(tmp_path: Path) -> Path:
    """Create a short stereo WAV file for testing."""
    path = tmp_path / "stereo.wav"
    waveform = torch.randn(2, 44100)
    torchaudio.save(str(path), waveform, 44100)
    return path


@pytest.fixture()
def toml_config(tmp_path: Path) -> Path:
    """Create a minimal TOML config file."""
    path = tmp_path / "chain.toml"
    path.write_text(
        '[[effects]]\nname = "gain"\ngain = 0.5\n\n' '[[effects]]\nname = "normalize"\npeak = 0.8\n'
    )
    return path


# ===================================================================
# Parsing tests
# ===================================================================


class TestCoerceValue:
    """Test value coercion utility."""

    def test_int(self) -> None:
        assert _coerce_value("42") == 42

    def test_float(self) -> None:
        assert _coerce_value("3.14") == pytest.approx(3.14)

    def test_bool_true(self) -> None:
        assert _coerce_value("true") is True
        assert _coerce_value("yes") is True

    def test_bool_false(self) -> None:
        assert _coerce_value("false") is False
        assert _coerce_value("no") is False

    def test_string_fallback(self) -> None:
        assert _coerce_value("lowpass") == "lowpass"


class TestParseEffectString:
    """Test --effect string parsing."""

    def test_name_only(self) -> None:
        fx = parse_effect_string("normalize")
        from torchfx.effect import Normalize

        assert isinstance(fx, Normalize)

    def test_single_positional(self) -> None:
        fx = parse_effect_string("gain:0.5")
        from torchfx.effect import Gain

        assert isinstance(fx, Gain)
        assert fx.gain == pytest.approx(0.5)

    def test_keyword_params(self) -> None:
        fx = parse_effect_string("reverb:decay=0.6,mix=0.3")
        from torchfx.effect import Reverb

        assert isinstance(fx, Reverb)
        assert fx.decay == pytest.approx(0.6)
        assert fx.mix == pytest.approx(0.3)

    def test_mixed_positional_and_keyword(self) -> None:
        fx = parse_effect_string("gain:0.8,gain_type=db")
        from torchfx.effect import Gain

        assert isinstance(fx, Gain)

    def test_filter_shortcut(self) -> None:
        fx = parse_effect_string("lowpass:cutoff=1000,q=0.707")
        from torchfx.filter.biquad import BiquadLPF

        assert isinstance(fx, BiquadLPF)
        assert fx.cutoff == pytest.approx(1000)

    def test_unknown_effect_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown effect"):
            parse_effect_string("doesnotexist:42")

    def test_too_many_positionals_raises(self) -> None:
        with pytest.raises(ValueError, match="Too many positional"):
            parse_effect_string("gain:0.5,amplitude,true,extra")

    def test_case_insensitive(self) -> None:
        fx = parse_effect_string("Normalize")
        from torchfx.effect import Normalize

        assert isinstance(fx, Normalize)


class TestParseEffectList:
    """Test parsing a list of effect specs."""

    def test_multiple_effects(self) -> None:
        effects = parse_effect_list(["gain:0.5", "normalize"])
        assert len(effects) == 2


class TestListEffects:
    """Test the effect listing helper."""

    def test_returns_sorted(self) -> None:
        names = list_effects()
        assert names == sorted(names)
        assert len(names) == len(EFFECT_REGISTRY)


# ===================================================================
# TOML config tests
# ===================================================================


class TestTomlConfig:
    """Test TOML configuration loading."""

    def test_load_effects(self, toml_config: Path) -> None:
        effects = load_effects_from_config(toml_config)
        assert len(effects) == 2

        from torchfx.effect import Gain, Normalize

        assert isinstance(effects[0], Gain)
        assert isinstance(effects[1], Normalize)

    def test_missing_name_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.toml"
        path.write_text("[[effects]]\npeak = 0.8\n")
        with pytest.raises(ValueError, match="must have a 'name'"):
            load_effects_from_config(path)

    def test_empty_effects_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.toml"
        path.write_text("[metadata]\ntitle = 'hello'\n")
        with pytest.raises(ValueError, match="No \\[\\[effects\\]\\]"):
            load_effects_from_config(path)

    def test_unknown_effect_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad2.toml"
        path.write_text('[[effects]]\nname = "doesnotexist"\n')
        with pytest.raises(ValueError, match="Unknown effect"):
            load_effects_from_config(path)


# ===================================================================
# CLI command tests (via CliRunner)
# ===================================================================


class TestVersionFlag:
    """Test --version flag."""

    def test_version(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "torchfx" in result.output


class TestInfoCommand:
    """Test the info subcommand."""

    def test_basic_info(self, wav_file: Path) -> None:
        result = runner.invoke(app, ["info", str(wav_file)])
        assert result.exit_code == 0
        assert "44,100" in result.output or "44100" in result.output  # sample rate
        assert "1" in result.output  # channels

    def test_stereo_info(self, stereo_wav: Path) -> None:
        result = runner.invoke(app, ["info", str(stereo_wav)])
        assert result.exit_code == 0
        assert "2" in result.output

    def test_file_not_found(self) -> None:
        result = runner.invoke(app, ["info", "/nonexistent/file.wav"])
        assert result.exit_code == 1


class TestProcessCommand:
    """Test the process subcommand."""

    def test_single_file(self, wav_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.wav"
        result = runner.invoke(
            app,
            ["process", str(wav_file), str(out), "-e", "gain:0.5"],
        )
        assert result.exit_code == 0
        assert out.exists()

    def test_single_file_with_config(
        self, wav_file: Path, tmp_path: Path, toml_config: Path
    ) -> None:
        out = tmp_path / "out.wav"
        result = runner.invoke(
            app,
            ["process", str(wav_file), str(out), "--config", str(toml_config)],
        )
        assert result.exit_code == 0
        assert out.exists()

    def test_batch_processing(self, tmp_path: Path) -> None:
        # Create multiple wav files
        for i in range(3):
            p = tmp_path / f"input_{i}.wav"
            torchaudio.save(str(p), torch.randn(1, 22050), 44100)

        out_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [
                "process",
                str(tmp_path / "input_*.wav"),
                "--output-dir",
                str(out_dir),
                "-e",
                "normalize",
            ],
        )
        assert result.exit_code == 0
        assert out_dir.exists()
        assert len(list(out_dir.glob("*.wav"))) == 3

    def test_no_effects_error(self, wav_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.wav"
        result = runner.invoke(app, ["process", str(wav_file), str(out)])
        assert result.exit_code == 1

    def test_no_matching_files(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["process", str(tmp_path / "*.nonexistent"), str(tmp_path / "out.wav"), "-e", "gain:1"],
        )
        assert result.exit_code == 1

    def test_multiple_effects(self, wav_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.wav"
        result = runner.invoke(
            app,
            [
                "process",
                str(wav_file),
                str(out),
                "-e",
                "gain:0.5",
                "-e",
                "normalize",
            ],
        )
        assert result.exit_code == 0
        assert out.exists()


class TestPlayCommand:
    """Test the play subcommand (mocked audio output)."""

    def test_file_not_found(self) -> None:
        result = runner.invoke(app, ["play", "/nonexistent/file.wav"])
        assert result.exit_code == 1

    @patch("torchfx.realtime._compat.get_sounddevice")
    def test_play_basic(self, mock_sd_getter: MagicMock, wav_file: Path) -> None:
        mock_sd = MagicMock()
        mock_sd_getter.return_value = mock_sd

        result = runner.invoke(app, ["play", str(wav_file)])
        assert result.exit_code == 0
        mock_sd.play.assert_called_once()
        mock_sd.wait.assert_called_once()


class TestRecordCommand:
    """Test the record subcommand (mocked audio input)."""

    @patch("torchfx.realtime._compat.get_sounddevice")
    def test_record_basic(self, mock_sd_getter: MagicMock, tmp_path: Path) -> None:
        import numpy as np

        mock_sd = MagicMock()
        mock_sd_getter.return_value = mock_sd
        # Simulate 1 second of recorded audio
        mock_sd.rec.return_value = np.random.randn(44100, 1).astype(np.float32)

        out = tmp_path / "rec.wav"
        result = runner.invoke(
            app,
            ["record", str(out), "--duration", "1"],
        )
        assert result.exit_code == 0
        assert out.exists()


# ===================================================================
# Effect registry completeness
# ===================================================================


class TestEffectRegistry:
    """Verify registry entries are instantiable."""

    @pytest.mark.parametrize(
        "name",
        [
            "normalize",
            "reverb",
        ],
    )
    def test_effect_default_instantiation(self, name: str) -> None:
        """Effects with sensible defaults can be constructed with no args."""
        cls, _ = EFFECT_REGISTRY[name]
        instance = cls()  # type: ignore[call-arg]
        assert instance is not None


# ===================================================================
# Phase 2: Sox-compatible commands
# ===================================================================


class TestConvertCommand:
    """Test the convert subcommand."""

    def test_basic_convert(self, wav_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.flac"
        result = runner.invoke(app, ["convert", str(wav_file), str(out)])
        assert result.exit_code == 0
        assert out.exists()

    def test_resample(self, wav_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "out_16k.wav"
        result = runner.invoke(app, ["convert", str(wav_file), str(out), "--rate", "16000"])
        assert result.exit_code == 0
        assert out.exists()
        w, sr = torchaudio.load(str(out))
        assert sr == 16000

    def test_channels(self, stereo_wav: Path, tmp_path: Path) -> None:
        out = tmp_path / "mono.wav"
        result = runner.invoke(app, ["convert", str(stereo_wav), str(out), "--channels", "1"])
        assert result.exit_code == 0
        w, sr = torchaudio.load(str(out))
        assert w.shape[0] == 1

    def test_file_not_found(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["convert", "/nonexistent.wav", str(tmp_path / "o.wav")])
        assert result.exit_code == 1


class TestTrimCommand:
    """Test the trim subcommand."""

    def test_trim_with_end(self, wav_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "clip.wav"
        result = runner.invoke(
            app, ["trim", str(wav_file), str(out), "--start", "0.0", "--end", "0.5"]
        )
        assert result.exit_code == 0
        w, sr = torchaudio.load(str(out))
        # Should be roughly 0.5s
        assert w.shape[1] == pytest.approx(sr * 0.5, abs=1)

    def test_trim_with_duration(self, wav_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "clip2.wav"
        result = runner.invoke(app, ["trim", str(wav_file), str(out), "--duration", "0.3"])
        assert result.exit_code == 0
        w, sr = torchaudio.load(str(out))
        assert w.shape[1] == pytest.approx(sr * 0.3, abs=1)

    def test_trim_file_not_found(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["trim", "/nonexistent.wav", str(tmp_path / "o.wav")])
        assert result.exit_code == 1

    def test_trim_bad_range(self, wav_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "bad.wav"
        result = runner.invoke(
            app, ["trim", str(wav_file), str(out), "--start", "5", "--end", "0.1"]
        )
        assert result.exit_code == 1


class TestConcatCommand:
    """Test the concat subcommand."""

    def test_concat_two_files(self, tmp_path: Path) -> None:
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        torchaudio.save(str(a), torch.randn(1, 22050), 44100)
        torchaudio.save(str(b), torch.randn(1, 22050), 44100)

        out = tmp_path / "combined.wav"
        result = runner.invoke(app, ["concat", str(a), str(b), "--output", str(out)])
        assert result.exit_code == 0
        w, sr = torchaudio.load(str(out))
        assert w.shape[1] == 44100  # 22050 + 22050

    def test_concat_mismatch_rate(self, tmp_path: Path) -> None:
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        torchaudio.save(str(a), torch.randn(1, 22050), 44100)
        torchaudio.save(str(b), torch.randn(1, 16000), 16000)

        out = tmp_path / "combined.wav"
        result = runner.invoke(app, ["concat", str(a), str(b), "--output", str(out)])
        assert result.exit_code == 1

    def test_concat_too_few_files(self, tmp_path: Path) -> None:
        a = tmp_path / "a.wav"
        torchaudio.save(str(a), torch.randn(1, 22050), 44100)

        out = tmp_path / "combined.wav"
        result = runner.invoke(app, ["concat", str(a), "--output", str(out)])
        assert result.exit_code == 1


class TestStatsCommand:
    """Test the stats subcommand."""

    def test_mono_stats(self, wav_file: Path) -> None:
        result = runner.invoke(app, ["stats", str(wav_file)])
        assert result.exit_code == 0
        assert "Peak" in result.output or "dBFS" in result.output

    def test_stereo_stats(self, stereo_wav: Path) -> None:
        result = runner.invoke(app, ["stats", str(stereo_wav)])
        assert result.exit_code == 0
        assert "Channel" in result.output

    def test_stats_file_not_found(self) -> None:
        result = runner.invoke(app, ["stats", "/nonexistent.wav"])
        assert result.exit_code == 1


# ===================================================================
# Phase 2: Preset management
# ===================================================================


class TestPresetCommands:
    """Test the preset sub-app."""

    def test_preset_list_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("cli.commands.preset.PRESETS_DIR", tmp_path / "presets")
        result = runner.invoke(app, ["preset", "list"])
        assert result.exit_code == 0
        assert "No presets" in result.output

    def test_preset_save_and_list(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        presets_dir = tmp_path / "presets"
        monkeypatch.setattr("cli.commands.preset.PRESETS_DIR", presets_dir)

        result = runner.invoke(
            app,
            ["preset", "save", "test-preset", "-e", "normalize", "-e", "gain:0.5"],
        )
        assert result.exit_code == 0
        assert "saved" in result.output.lower()
        assert (presets_dir / "test-preset.toml").exists()

        # Verify list shows it
        result = runner.invoke(app, ["preset", "list"])
        assert result.exit_code == 0
        assert "test-preset" in result.output

    def test_preset_save_no_effects(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("cli.commands.preset.PRESETS_DIR", tmp_path / "presets")
        result = runner.invoke(app, ["preset", "save", "empty"])
        assert result.exit_code == 1

    def test_preset_show(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        (presets_dir / "demo.toml").write_text('[[effects]]\nname = "normalize"\n')
        monkeypatch.setattr("cli.commands.preset.PRESETS_DIR", presets_dir)
        result = runner.invoke(app, ["preset", "show", "demo"])
        assert result.exit_code == 0
        assert "normalize" in result.output

    def test_preset_show_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("cli.commands.preset.PRESETS_DIR", tmp_path / "presets")
        result = runner.invoke(app, ["preset", "show", "nonexistent"])
        assert result.exit_code == 1

    def test_preset_delete(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        (presets_dir / "todel.toml").write_text('[[effects]]\nname = "normalize"\n')
        monkeypatch.setattr("cli.commands.preset.PRESETS_DIR", presets_dir)

        result = runner.invoke(app, ["preset", "delete", "todel"])
        assert result.exit_code == 0
        assert "deleted" in result.output.lower()
        assert not (presets_dir / "todel.toml").exists()

    def test_preset_delete_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("cli.commands.preset.PRESETS_DIR", tmp_path / "presets")
        result = runner.invoke(app, ["preset", "delete", "nope"])
        assert result.exit_code == 1

    def test_preset_save_overwrite_blocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        (presets_dir / "dup.toml").write_text('[[effects]]\nname = "normalize"\n')
        monkeypatch.setattr("cli.commands.preset.PRESETS_DIR", presets_dir)

        result = runner.invoke(app, ["preset", "save", "dup", "-e", "normalize"])
        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_preset_save_force_overwrite(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        (presets_dir / "dup.toml").write_text('[[effects]]\nname = "normalize"\n')
        monkeypatch.setattr("cli.commands.preset.PRESETS_DIR", presets_dir)

        result = runner.invoke(app, ["preset", "save", "dup", "-e", "gain:0.5", "--force"])
        assert result.exit_code == 0

    def test_preset_apply(
        self, wav_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        (presets_dir / "testfx.toml").write_text('[[effects]]\nname = "gain"\ngain = 0.5\n')
        monkeypatch.setattr("cli.commands.preset.PRESETS_DIR", presets_dir)

        out = tmp_path / "applied.wav"
        result = runner.invoke(app, ["preset", "apply", "testfx", str(wav_file), str(out)])
        assert result.exit_code == 0
        assert out.exists()


# ===================================================================
# Phase 2: REPL commands (unit-level)
# ===================================================================


class TestReplCommands:
    """Test individual REPL command handlers."""

    def test_cmd_add_and_list(self) -> None:
        from cli.repl import _ReplState, _cmd_add, _cmd_list

        state = _ReplState()
        result = _cmd_add(state, ["normalize"])
        assert "Added" in result
        assert len(state.effects) == 1

        result = _cmd_list(state, [])
        assert "normalize" in result

    def test_cmd_add_invalid(self) -> None:
        from cli.repl import _ReplState, _cmd_add

        state = _ReplState()
        result = _cmd_add(state, ["doesnotexist"])
        assert "Unknown effect" in result

    def test_cmd_remove(self) -> None:
        from cli.repl import _ReplState, _cmd_add, _cmd_remove

        state = _ReplState()
        _cmd_add(state, ["normalize"])
        _cmd_add(state, ["gain:0.5"])

        result = _cmd_remove(state, ["1"])
        assert "Removed" in result
        assert len(state.effects) == 1

    def test_cmd_remove_invalid_index(self) -> None:
        from cli.repl import _ReplState, _cmd_remove

        state = _ReplState()
        result = _cmd_remove(state, ["5"])
        assert "Invalid" in result

    def test_cmd_effects(self) -> None:
        from cli.repl import _ReplState, _cmd_effects

        state = _ReplState()
        result = _cmd_effects(state, [])
        assert "Available effects" in result
        assert "gain" in result

    def test_cmd_clear(self) -> None:
        from cli.repl import _ReplState, _cmd_add, _cmd_clear

        state = _ReplState()
        _cmd_add(state, ["normalize"])
        _cmd_add(state, ["gain:0.5"])

        result = _cmd_clear(state, [])
        assert "Cleared" in result
        assert len(state.effects) == 0

    def test_cmd_info_no_file(self) -> None:
        from cli.repl import _ReplState, _cmd_info

        state = _ReplState()
        result = _cmd_info(state, [])
        assert "No file loaded" in result

    def test_cmd_load(self, wav_file: Path) -> None:
        from cli.repl import _ReplState, _cmd_load

        state = _ReplState()
        result = _cmd_load(state, [str(wav_file)])
        assert "Loaded" in result
        assert state.wave is not None

    def test_cmd_load_not_found(self) -> None:
        from cli.repl import _ReplState, _cmd_load

        state = _ReplState()
        result = _cmd_load(state, ["/nonexistent.wav"])
        assert "not found" in result

    def test_cmd_save(self, wav_file: Path, tmp_path: Path) -> None:
        from cli.repl import _ReplState, _cmd_add, _cmd_load, _cmd_save

        state = _ReplState()
        _cmd_load(state, [str(wav_file)])
        _cmd_add(state, ["normalize"])

        out = tmp_path / "repl_out.wav"
        result = _cmd_save(state, [str(out)])
        assert "Saved" in result
        assert out.exists()

    def test_cmd_save_no_file(self) -> None:
        from cli.repl import _ReplState, _cmd_save

        state = _ReplState()
        result = _cmd_save(state, ["out.wav"])
        assert "No file loaded" in result


# ===================================================================
# Phase 2: Watch command
# ===================================================================


class TestWatchCommand:
    """Test the watch subcommand basics."""

    def test_watch_missing_dir(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "watch",
                str(tmp_path / "nonexistent"),
                str(tmp_path / "out"),
                "-e",
                "normalize",
            ],
        )
        assert result.exit_code == 1

    def test_watch_no_effects(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        result = runner.invoke(app, ["watch", str(src), str(tmp_path / "out")])
        assert result.exit_code == 1
