from typing import Union


class ColorHelper:
    """
    Helper class meant to streamline the use of colors for several purposes. Enjoy.
    """

    def _convert_hex_to_rgb(self, hex_color: str) -> tuple:
        """
        Converts a hex color to a tuple of RGB values.

        NOTE: This might not always be accurate.
        """
        return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))

    def colorize_string(self, string: str, text_color_hex: str, background_color_hex: Union[str, None] = None) -> str:
        """
        Colorizes a string with a given color.
        """
        string = string

        if background_color_hex is not None:
            background_color_rgb = self._convert_hex_to_rgb(background_color_hex)
            closest_ansi_color_code_background = f"\x1b[48;2;{background_color_rgb[0]};{background_color_rgb[1]};{background_color_rgb[2]}m"
            string = f"{closest_ansi_color_code_background}{string}"

        text_color_rgb = self._convert_hex_to_rgb(text_color_hex)
        closest_ansi_color_code = f"\x1b[38;2;{text_color_rgb[0]};{text_color_rgb[1]};{text_color_rgb[2]}m"
        string = f"{closest_ansi_color_code}{string}"

        return string
