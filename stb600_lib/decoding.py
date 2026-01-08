"""
Decoding Module - Value Decoding from ROI Counts
================================================

Functions for decoding numeric values from ROI label counts.
"""


def decode_roi_to_number(counts):
    """
    Decode ROI label counts to a single digit (0-9).

    Encoding scheme:
        Number 0 = 1 small
        Number 1 = 2 small
        Number 2 = 3 small
        Number 3 = 4 small
        Number 4 = 1 small + 1 medium
        Number 5 = 2 small + 1 medium
        Number 6 = 3 small + 1 medium
        Number 7 = 1 small + 1 large
        Number 8 = 2 small + 1 large
        Number 9 = 3 small + 1 large

    Parameters
    ----------
    counts : dict
        Dictionary with keys "small", "medium", "large" containing counts.

    Returns
    -------
    int
        Decoded digit (0-9).

    Raises
    ------
    ValueError
        If counts represent an invalid digit combination.
    """
    small = counts.get("small", 0)
    medium = counts.get("medium", 0)
    big = counts.get("large", 0)

    # Basic validity rules
    if small < 0 or medium < 0 or big < 0:
        raise ValueError(f"Negative counts are not allowed: {counts}")
    if medium > 1 or big > 1:
        raise ValueError(f"Too many MEDIUM or BIG parts: {counts}")

    # Only small: 1S->0, 2S->1, 3S->2, 4S->3
    if medium == 0 and big == 0:
        if 1 <= small <= 4:
            return small - 1
        else:
            raise ValueError(f"Invalid SMALL count for pure-small digit: {counts}")

    # 1 medium, no big: 1S+1M -> 4, 2S+1M -> 5, 3S+1M -> 6
    if medium == 1 and big == 0:
        if 1 <= small <= 3:
            return small + 3
        else:
            raise ValueError(f"Invalid SMALL count for medium-digit: {counts}")

    # 1 big, no medium: 1S+1B -> 7, 2S+1B -> 8, 3S+1B -> 9
    if big == 1 and medium == 0:
        if 1 <= small <= 3:
            return small + 6
        else:
            raise ValueError(f"Invalid SMALL count for big-digit: {counts}")

    # Any other combination is invalid
    raise ValueError(f"Invalid combination for a digit: {counts}")


def digits_to_int(digits):
    """
    Convert a list of digits to an integer.

    Parameters
    ----------
    digits : list of int or None
        List of digits [d1, d2, d3] to form integer 123.
        None values are ignored.

    Returns
    -------
    int or None
        Integer formed from valid digits, or None if no valid digits.
    """
    valid = [d for d in digits if d is not None]
    if not valid:
        return None
    return int("".join(str(d) for d in valid))


def compute_total_value_from_rois(decoded_digits, piece_color):
    """
    Compute the total value from decoded ROI digits based on piece color.

    Rules by piece color:
      - 'red'     -> ROI0 * ROI1 (multiplication)
      - 'yellow'  -> ROI0ROI1ROI2 * 10 (concatenated digits times 10)
      - 'blue'    -> ROI0ROI1ROI2 (concatenated digits)

    Parameters
    ----------
    decoded_digits : list of int or None
        List of decoded digits from ROIs [ROI0, ROI1, ...].
    piece_color : str or None
        Piece color determining the calculation rule.

    Returns
    -------
    int or None
        Computed total value, or None if calculation not possible.
    """
    color = (piece_color or "").lower()

    # Red: ROI0 * ROI1
    if color == "red":
        if len(decoded_digits) < 2 or None in decoded_digits[:2]:
            return None
        return decoded_digits[0] * decoded_digits[1]

    # Yellow / Blue: form number from all valid digits
    number = digits_to_int(decoded_digits)
    if number is None:
        return None

    if color == "yellow":
        return number * 10
    elif color == "blue":
        return number

    # Unknown color: return number as-is
    return number
