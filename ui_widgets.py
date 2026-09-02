"""
ui_widgets.py — Reusable UI Drawing Primitives for Apple-Tesla Glassmorphic HUD
================================================================================
Provides:
  - Vertical Gradient Surface Fills.
  - Frosted Glass Panels with Specular Top Edge Highlights.
  - Multi-Layer Soft Glow Outlines for Critical Safety Alerts (FCW/TTC).
  - Live Color-Coded Performance FPS Meter.
"""

import pygame
import config as cfg


def draw_gradient_rect(
    surface: pygame.Surface,
    rect: pygame.Rect,
    top_color: tuple[int, int, int],
    bottom_color: tuple[int, int, int],
    border_radius: int = 0
):
    """Fills a rectangle with a smooth vertical linear gradient."""
    grad_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    h = max(1, rect.height)
    for y in range(h):
        ratio = y / float(h)
        r = int(top_color[0] + (bottom_color[0] - top_color[0]) * ratio)
        g = int(top_color[1] + (bottom_color[1] - top_color[1]) * ratio)
        b = int(top_color[2] + (bottom_color[2] - top_color[2]) * ratio)
        pygame.draw.line(grad_surf, (r, g, b, 255), (0, y), (rect.width, y))

    if border_radius > 0:
        mask_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(mask_surf, (255, 255, 255, 255), (0, 0, rect.width, rect.height), border_radius=border_radius)
        grad_surf.blit(mask_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

    surface.blit(grad_surf, (rect.x, rect.y))


def draw_glass_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    border_radius: int = 6,
    border_color: tuple[int, int, int] = cfg.COLOR_BORDER_THIN,
    top_color: tuple[int, int, int] = cfg.COLOR_PANEL_BG,
    bot_color: tuple[int, int, int] = cfg.COLOR_PANEL_BG_LO
):
    """
    Renders a frosted glass panel with a vertical gradient, subtle 1px border,
    and a delicate 1px specular highlight along the top edge.
    """
    draw_gradient_rect(surface, rect, top_color, bot_color, border_radius=border_radius)
    pygame.draw.rect(surface, border_color, rect, 1, border_radius=border_radius)

    # Specular Top Highlight Line (simulates light refraction across glass edge)
    if rect.width > 20:
        hl_y = rect.y + 1
        hl_start_x = rect.x + border_radius
        hl_end_x = rect.x + rect.width - border_radius
        pygame.draw.line(surface, cfg.COLOR_BORDER_HL, (hl_start_x, hl_y), (hl_end_x, hl_y), 1)


def draw_glow_rect(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: tuple[int, int, int],
    border_radius: int = 4,
    strength: int = 3
):
    """
    Renders soft expanding, fading glow outlines around a target rectangle
    for active critical warnings (e.g. Tesla Red FCW alerts).
    """
    glow_surf = pygame.Surface((rect.width + strength * 8, rect.height + strength * 8), pygame.SRCALPHA)
    for i in range(strength, 0, -1):
        alpha = int(45 * (1.0 - (i / float(strength + 1))))
        g_col = (color[0], color[1], color[2], alpha)
        g_rect = pygame.Rect(
            strength * 4 - i * 2,
            strength * 4 - i * 2,
            rect.width + i * 4,
            rect.height + i * 4
        )
        pygame.draw.rect(glow_surf, g_col, g_rect, 2, border_radius=border_radius + i)

    surface.blit(glow_surf, (rect.x - strength * 4, rect.y - strength * 4))


def fps_color(fps: float) -> tuple[int, int, int]:
    """Returns dynamic color based on live frame rate performance."""
    if fps >= 55.0:
        return cfg.COLOR_APPLE_GREEN
    elif fps >= 30.0:
        return cfg.COLOR_APPLE_AMBER
    return cfg.COLOR_TESLA_RED


def draw_fps_hud(
    surface: pygame.Surface,
    font: pygame.font.Font,
    fps: float,
    pos: tuple[int, int]
):
    """Draws a live, color-coded FPS reading at the given coordinates."""
    col = fps_color(fps)
    txt_surf = font.render(f"FPS: {fps:04.1f}", True, col)
    surface.blit(txt_surf, pos)
