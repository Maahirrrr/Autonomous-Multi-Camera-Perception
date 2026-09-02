"""
config.py — Central Configuration & Runtime Settings for L4 Perception Cockpit
==============================================================================
Provides:
  - Apple-Tesla Theme Color Palette Constants.
  - Display & Simulation Dimension Constants.
  - Settings Dataclass with CLI Argument Parsing.
  - Keybinding Help Documentation.
"""

from dataclasses import dataclass
import argparse
from typing import Optional


# =====================================================================
# APPLE-TESLA OBSIDIAN GLASS THEME COLOR PALETTE
# =====================================================================
COLOR_BG_PURE = (8, 9, 12)           # #08090C Frosted Obsidian Base
COLOR_PANEL_BG = (14, 17, 23)        # #0E1117 Deep Titanium Glass (Top)
COLOR_PANEL_BG_LO = (11, 13, 18)     # #0B0D12 Deep Titanium Glass (Bottom Gradient)
COLOR_CARD_BG = (20, 24, 34)         # #141822 Translucent Glass Card
COLOR_BORDER_THIN = (34, 40, 54)     # #222836 1px Precision Chamfer
COLOR_BORDER_HL = (65, 78, 105)      # Specular Glass Top Highlight
COLOR_TEXT_MAIN = (245, 248, 255)    # #F5F8FF Pure Crisp White
COLOR_TEXT_MUTED = (138, 146, 162)   # #8A92A2 Apple Neutral Gray
COLOR_TESLA_CYAN = (0, 229, 255)     # #00E5FF Electric Tesla Cyan
COLOR_TESLA_RED = (255, 51, 75)      # #FF334B Ultra Red Alert
COLOR_APPLE_GREEN = (48, 209, 88)    # #30D158 Apple iOS Emerald
COLOR_APPLE_AMBER = (255, 214, 10)   # #FFD60A Apple iOS Amber

# =====================================================================
# DISPLAY & SIMULATION RESOLUTIONS
# =====================================================================
LOGICAL_WIDTH = 1280
LOGICAL_HEIGHT = 800
DEFAULT_FPS = 60
DEFAULT_MAX_EXPORT_FRAMES = 300

KEY_HELP = "[TAB] Pilot | [N] Night | [P] Weather | [R] Randomize | [A/D] Lane Change | [W/S] Speed"


@dataclass
class Settings:
    """Runtime configuration settings for the Autonomous Perception Cockpit."""
    width: int = LOGICAL_WIDTH
    height: int = LOGICAL_HEIGHT
    fullscreen: bool = False
    fps: int = DEFAULT_FPS
    seed: Optional[int] = None
    export_path: Optional[str] = None
    max_export_frames: int = DEFAULT_MAX_EXPORT_FRAMES
    log_level: str = "INFO"
    show_fps: bool = True


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds and returns the comprehensive command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Tesla Level 4 Autonomous 360° Perception Cockpit • Apple-Tesla Design DNA"
    )
    parser.add_argument("--fullscreen", action="store_true", help="Launch in fullscreen mode")
    parser.add_argument("--width", type=int, default=LOGICAL_WIDTH, help="Window display width in pixels")
    parser.add_argument("--height", type=int, default=LOGICAL_HEIGHT, help="Window display height in pixels")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Target simulation frame rate (default: 60)")
    parser.add_argument("--seed", type=int, default=None, help="Fixed random seed for reproducible scenarios")
    parser.add_argument("--export", type=str, default=None, help="Export cockpit session to MP4 video path")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_EXPORT_FRAMES, help="Max frames for video export")
    parser.add_argument("--no-fps-hud", action="store_true", help="Hide the live FPS readout in the header")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity level"
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Instantiates a Settings dataclass instance from parsed command-line arguments."""
    return Settings(
        width=args.width,
        height=args.height,
        fullscreen=args.fullscreen,
        fps=args.fps,
        seed=args.seed,
        export_path=args.export,
        max_export_frames=args.max_frames,
        log_level=args.log_level,
        show_fps=not args.no_fps_hud
    )


def parse_args(argv: Optional[list[str]] = None) -> Settings:
    """Convenience helper to parse command-line arguments and return Settings instance."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return settings_from_args(args)
