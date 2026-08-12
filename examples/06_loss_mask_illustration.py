'''Show how loss masking excludes user tokens.'''


def build_loss_mask(input_ids: list[int], response_start_idx: int) -> list[int]:
    '''Returns a mask where response tokens have 1, others have 0.'''
    mask = [0] * len(input_ids)
    for i in range(response_start_idx, len(input_ids)):
        mask[i] = 1
    return mask


if __name__ == "__main__":
    # Fake token IDs for illustration
    input_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # e.g., "system user assistant..."
    response_start = 6  # assistant tokens start here
    mask = build_loss_mask(input_ids, response_start)
    print("Input IDs:  ", input_ids)
    print("Loss mask:  ", mask)
    print("Only tokens where mask=1 contribute to loss.")
