import itertools
from collections import Counter

import midi
import numpy as np

_darezzo = ['do', 'do#', 're', 're#', 'mi', 'fa', 'fa#', 'sol', 'sol#', 'la', 'la#', 'si'] * 8


def _pitch2note(pitch):
    return _darezzo[pitch - 72] + str(pitch // 12 - 2)


_piano_order = ['la', 'la#', 'si', 'do', 'do#', 're', 're#', 'mi', 'fa', 'fa#', 'sol', 'sol#']
_note2index = {_piano_order[i % 12] + str((i - 3) // 12): i for i in range(88)}


def _pitch2index(pitch):
    return pitch - 21  # 21 is MIDI A0


def _index2pitch(index):
    return index + 21  # max pitch is 108, MIDI C8


def _notes2vector(notes):
    keyboard = np.zeros((88), int)
    for note in notes:
        keyboard[_note2index[note]] = 1
    return keyboard


def midi2numpy(midifile, outfile=None):
    # read the midi file
    m = midi.read_midifile(midifile)

    notes = {}
    for i, track in enumerate(m):
        tick_offset = 0
        notes[i] = []
        for event in track:
            tick_offset += event.tick
            if isinstance(event, midi.NoteOnEvent):
                pitch = event.pitch
                velocity = event.velocity
                if velocity > 0:
                    notes[i].append((pitch, tick_offset))

    # merge tracks
    notes = sorted(list(itertools.chain.from_iterable(list(notes.values()))), key=lambda x: x[1])
    diff_tick = [x for x in [notes[i][1] - notes[i - 1][1] for i in range(1, len(notes))] if x > 0]
    min_tick = Counter(diff_tick).most_common(1)[0][0]  # min(diff_tick)

    first_note = min([tick for _, tick in notes])

    # print('min_tick', min_tick)
    # normalize everything subtracting the first tick and then dividing by min_tick
    notes = [(_pitch2note(pitch), (tick - first_note) // min_tick) for pitch, tick in notes]
    # print(notes)

    _, last_tick = notes[-1]

    tick_note = {}
    for pitch, tick in notes:
        if tick not in tick_note: tick_note[tick] = []
        tick_note[tick].append(pitch)

    time_span = []
    for t in range(last_tick + 1):
        if t in tick_note:
            time_span.append(sorted(tick_note[t]))
        elif t > 0:
            time_span.append(time_span[t - 1])

    # print('time-span')
    # for time in time_span:
    #    print(time)

    # print('number of parts ', len(time_span)//8)

    X = np.vstack([_notes2vector(time_span_i) for time_span_i in time_span])

    if outfile:
        np.save(outfile, X)

    return X


def numpy2midi(X, midifile):
    # find for each timestep the set of on-notes
    X = [set(np.nonzero(x)[0]) for x in X]

    # prepend and append an empty step to the song
    X.insert(0, set())
    X.append(set())

    track = midi.Track()
    time_sig = midi.TimeSignatureEvent(tick=0, numerator=2, denominator=4, metronome=24, thirtyseconds=8)
    track.append(time_sig)

    tick = 0
    for previous, current in zip(X[:-1], X[1:]):
        notes_to_off = previous - current  # notes in the previous but not in the current (ending notes)
        for pitch in map(_index2pitch, notes_to_off):
            off = midi.NoteOffEvent(tick=tick, pitch=pitch, velocity=0)  # velocity is ignored in NoteOffEvents
            track.append(off)
            tick = 0

        notes_to_on = current - previous  # notes in the current but not in the previous (beginning notes)
        for pitch in map(_index2pitch, notes_to_on):
            on = midi.NoteOnEvent(tick=tick, pitch=pitch, velocity=100)
            track.append(on)
            tick = 0

        tick += 120

    # end of track event
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    # instantiate a MIDI Pattern (contains a list of tracks)
    pattern = midi.Pattern(resolution=96)
    # append the track to the pattern
    pattern.append(track)
    # save the pattern to disk
    midi.write_midifile(midifile, pattern)


if __name__ == '__main__':
    # midi2numpy('midi/sunflower.mid') # ,'sunflower.npy')
    # numpy2midi(midi2numpy('midi/Joplin/BreezeFromAlabama2.mid'), 'prova.mid')
    numpy2midi(np.eye(88), 'cromatica.mid')
    # numpy2midi(np.load('sunflower.npy'), 'sunflower_reconstructed.mid')
