"""Legacy compatibility wrapper. The supported renderer is cad_renderer_cpu."""
from cad_renderer_cpu import render_step_svg


def render_step_view(file_path, output_path, view_direction, up_direction, **kwargs):
    return render_step_svg(
        file_path=file_path,
        output_path=output_path,
        view_direction=view_direction,
        up_direction=up_direction,
        **kwargs
    )
