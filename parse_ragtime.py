import midi
import itertools
import numpy as np

_darezzo = ['do', 'do#', 're', 're#', 'mi', 'fa', 'fa#', 'sol', 'sol#', 'la', 'la#', 'si'] * 8
def _pitch2note(pitch):
    return _darezzo[pitch - 72] + str(pitch // 12 - 2)

_piano_order=['la', 'la#', 'si', 'do', 'do#', 're', 're#', 'mi', 'fa', 'fa#', 'sol', 'sol#']
_note2index = {_piano_order[i % 12] + str((i - 3) // 12):i for i in range(88)}

def _notes2vector(notes):
    keyboard = np.zeros((88), int)
    for note in notes:
        keyboard[_note2index[note]] = 1
    return keyboard

def midi2numpy(midifile, outfile=None):
    #read the midi file
    m = midi.read_midifile(midifile)

    notes = {}
    for i,track in enumerate(m):
        tick_offset = 0
        notes[i]=[]
        for event in track:
            tick_offset += event.tick
            if isinstance(event, midi.NoteOnEvent):
                pitch = event.pitch
                velocity = event.velocity
                if velocity>0:
                    notes[i].append((pitch,tick_offset))

    # merge tracks
    notes = sorted(list(itertools.chain.from_iterable(list(notes.values()))), key=lambda x:x[1])
    min_tick = min([x for x in [notes[i][1] - notes[i-1][1] for i in range(1,len(notes))] if x>0])
    first_note = min([tick for _,tick in notes])

    print('min_tick',min_tick)
    #normalize everything subtracting the first tick and then dividing by min_tick
    notes = [(_pitch2note(pitch), (tick - first_note) // min_tick) for pitch, tick in notes]
    print(notes)

    _,last_tick = notes[-1]

    tick_note = {}
    for pitch,tick in notes:
        if tick not in tick_note: tick_note[tick]=[]
        tick_note[tick].append(pitch)

    time_span = []
    for t in range(last_tick+1):
        if t in tick_note:
            time_span.append(sorted(tick_note[t]))
        elif t>0:
            time_span.append(time_span[t-1])

    print('time-span')
    for time in time_span:
        print(time)

    # print('number of parts ', len(time_span)//8)

    X = np.vstack([_notes2vector(time_span_i) for time_span_i in time_span])

    if outfile:
        np.save(outfile,X)

    return X

midi2numpy('midi/sunflower.mid','np/sunflower.npy')