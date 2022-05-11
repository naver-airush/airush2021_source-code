from jamo import j2h, j2hcj, get_jamo_class, is_jamo


def read_strings(input_file):
    return open(input_file, 'r', encoding='utf-8').read().splitlines()


def write_strings(output_file, data):
    with open(output_file, 'w', encoding='utf-8') as f:
        for x in data:
            f.write(str(x) + '\n')


def reconstruct_jamo(decomposed, debug=False, remove_incomplete=True):
    reconstructed = []
    current_char = []
    current_state = 'init'  # init, lead, vowel
    for c in decomposed:
        if is_jamo(c):
            try:
                jamo_class = get_jamo_class(c)
            except:  # isolated
                reconstructed.append(j2hcj(c))
                continue
            if jamo_class == 'lead':
                if current_state == 'init':
                    assert len(current_char) == 0
                    current_char.append(c)
                    current_state = 'lead'
                elif current_state == 'lead':
                    assert len(current_char) == 1
                    if not remove_incomplete:
                        reconstructed.append(j2hcj(current_char[0]))
                    current_char = [c]
                    current_state = 'lead'
                elif current_state == 'vowel':
                    assert len(current_char) == 2
                    reconstructed.append(j2h(*current_char))
                    current_char = [c]
                    current_state = 'lead'

            elif jamo_class == 'vowel':
                if current_state == 'init':
                    assert len(current_char) == 0
                    if not remove_incomplete:
                        reconstructed.append(j2hcj(c))
                elif current_state == 'lead':
                    assert len(current_char) == 1
                    current_char.append(c)
                    current_state = 'vowel'
                elif current_state == 'vowel':
                    assert len(current_char) == 2
                    reconstructed.append(j2h(*current_char))
                    if not remove_incomplete:
                        reconstructed.append(j2hcj(c))
                    current_char = []
                    current_state = 'init'
            else:  # jongsung
                if current_state == 'init':
                    assert len(current_char) == 0
                    if not remove_incomplete:
                        reconstructed.append(j2hcj(c))
                elif current_state == 'lead':
                    assert len(current_char) == 1
                    if not remove_incomplete:
                        reconstructed.append(j2hcj(current_char[0]))
                        reconstructed.append(j2hcj(c))
                elif current_state == 'vowel':
                    assert len(current_char) == 2
                    current_char.append(c)
                    reconstructed.append(j2h(*current_char))

                current_char = []
                current_state = 'init'

        else:
            if current_state == 'init':
                assert len(current_char) == 0
            if current_state == 'lead':
                assert len(current_char) == 1
                if not remove_incomplete:
                    reconstructed.append(j2hcj(current_char[0]))
            elif current_state == 'vowel':
                assert len(current_char) == 2
                reconstructed.append(j2h(*current_char))

            reconstructed.append(c)
            current_char = []
            current_state = 'init'

        if debug:
            print(current_state, c, current_char, reconstructed)

    # if there is leftover
    if len(current_char) > 0:
        if current_state == 'lead':
            assert len(current_char) == 1
            if not remove_incomplete:
                reconstructed.append(j2hcj(current_char[0]))
        elif current_state == 'vowel':
            assert len(current_char) == 2
            reconstructed.append(j2h(*current_char))

    if debug:
        print(current_state, c, current_char, reconstructed)

    return ''.join(reconstructed)
