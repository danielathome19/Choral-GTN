import os
import sys
import copy
import time
import json
# import torch
import random
import warnings
import itertools
import traceback
import subprocess
import multiprocessing
import tensorflow as tf
from tqdm import tqdm
from p_tqdm import p_uimap
from functools import partial
from chorder import Dechorder
from collections import Counter
from more_itertools import split_before
from miditoolkit.midi.parser import MidiFile
from miditoolkit.midi.containers import Instrument
from miditoolkit.midi.containers import Note as mtkNote


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# region Encoding
pit2alphabet = ['C', 'd', 'D', 'e', 'E', 'F', 'g', 'G', 'a', 'A', 'b', 'B']
char2pit = {x: id for id, x in enumerate(pit2alphabet)}


def pit2str(x):
    octave = x // 12
    octave = octave - 1 if octave > 0 else 'O'
    rel_pit = x % 12
    return pit2alphabet[rel_pit] + str(octave)


def str2pit(x):
    rel_pit = char2pit[x[0]]
    octave = (int(x[1]) if x[1] != 'O' else -1) + 1
    return octave * 12 + rel_pit


def int2char(x):
    if x <= 9:
        return str(x)
    if x <= 35:
        return chr(ord('a') + (x - 10))
    if x < 62:
        return chr(ord('A') + (x - 36))
    assert False, f'invalid number {x}'


def char2int(c):
    num = ord(c)
    A, a, Z, z = ord('A'), ord('a'), ord('Z'), ord('z')
    if a <= num <= z:
        return 10 + num - a
    elif A <= num <= Z:
        return 36 + num - A
    elif ord('0') <= num <= ord('9'):
        return num - ord('0')
    assert False, f'invalid character {c}'


def pos2str(ons):
    if ons < 62:
        return 'p' + int2char(ons)
    return 'P' + int2char(ons - 62)


def bom2str(ons):
    if ons < 62:
        return 'm' + int2char(ons)
    return 'M' + int2char(ons - 62)


def dur2str(ons):
    if ons < 62:
        return 'r' + int2char(ons)
    return 'R' + int2char(ons - 62)


def trk2str(ons):
    if ons < 62:
        return 't' + int2char(ons)
    return 'T' + int2char(ons - 62)


def ins2str(ons):  # 0 - 128
    if ons < 62:
        return 'x' + int2char(ons)
    ons -= 62
    if ons < 62:
        return 'X' + int2char(ons)
    ons -= 62
    if ons < 62:
        return 'y' + int2char(ons)
    return 'Y' + int2char(ons - 62)


def ispitch(x):  # judge if a event str is a pitch (CO - B9)
    return len(x) == 2 and x[0] in char2pit and (x[1] == 'O' or x[1].isdigit())


def ison(x):  # judge if a event str is a bpe token
    if len(x) % 2 != 0 or len(x) < 2:
        return False
    for i in range(0, len(x), 2):
        if not ispitch(x[i:i + 2]):
            return False

    return True


def bpe_str2int(x):
    if len(x) == 2:
        return 0, str2pit(x)
    res = []
    for i in range(0, len(x), 2):
        res.append(str2pit(x[i:i + 2]))
    return (1,) + tuple(sorted(res))


def sort_tok_str(x):
    c = x[0].lower()
    if c in ('r', 't', 'x', 'y'):
        #         if x in ('RZ', 'TZ', 'YZ'):
        #             return (c if c != 'y' else 'x', False, -1)
        return c, not x[0].islower(), char2int(x[1])
    if c in ('m', 'p'):
        return c, not x[0].islower(), char2int(x[1])

    if c == 'h':
        return c, char2pit[x[1]] if x[1] != 'N' else 12, x[2:]
    if c == 'n':
        return 'w', x
    if ison(x):
        return ('a',) + bpe_str2int(x)

    return 'A', x[1] != 'b', x[1] != 'p', x[1] != 'e'

# endregion Encoding


# region Preprocess_MIDI
WORKERS = 32


def measure_calc_chord(evt_seq):
    assert evt_seq[0][1] == 'BOM', "wrong measure for chord"
    bom_tick = evt_seq[0][0]
    ts = min(evt_seq[0][-1], 8)
    chroma = Counter()
    mtknotes = []
    for evt in evt_seq[1:-1]:
        assert evt[1] == 'ON', "wrong measure for chord: " + evt[1] + evt_seq[-1][1]
        if evt[3] == 128:  # exclude drums
            continue
        o, p, d = evt[0] - bom_tick, evt[2], evt[-1]
        if p < 21 or p > 108:  # exclude unusual pitch
            continue
        if o < 8:
            note = mtkNote(60, p, o, o + d if o > 0 else 8)
            mtknotes.append(note)
        else:
            break

    chord, score = Dechorder.get_chord_quality(mtknotes, start=0, end=ts)
    if score < 0:
        return [bom_tick, 'CHR', None, None, None, None, 'NA']
    return [bom_tick, 'CHR', None, None, None, None,
            pit2alphabet[chord.root_pc] + (chord.quality if chord.quality != '7' else 'D7')]


def merge_drums(p_midi):  # merge all percussions
    drum_0_lst = []
    new_instruments = []
    for instrument in p_midi.instruments:
        if not len(instrument.notes) == 0:
            # --------------------
            if instrument.is_drum:
                for note in instrument.notes:
                    drum_0_lst.append(note)
            else:
                new_instruments.append(instrument)
    if len(drum_0_lst) > 0:
        drum_0_lst.sort(key=lambda x: x.start)
        # remove duplicate
        drum_0_lst = list(k for k, _ in itertools.groupby(drum_0_lst))

        drum_0_instrument = Instrument(program=0, is_drum=True, name="drum")
        drum_0_instrument.notes = drum_0_lst
        new_instruments.append(drum_0_instrument)

    p_midi.instruments = new_instruments


def merge_sparse_track(p_midi, CANDI_THRES=50, MIN_THRES=5):  # merge track has too less notes
    good_instruments = []
    bad_instruments = []
    good_instruments_idx = []
    for instrument in p_midi.instruments:
        if len(instrument.notes) < CANDI_THRES:
            bad_instruments.append(instrument)
        else:
            good_instruments.append(instrument)
            good_instruments_idx.append((instrument.program, instrument.is_drum))

    for bad_instrument in bad_instruments:
        if (bad_instrument.program, bad_instrument.is_drum) in good_instruments_idx:
            # find one track to merge
            for instrument in good_instruments:
                if bad_instrument.program == instrument.program and \
                        bad_instrument.is_drum == instrument.is_drum:
                    instrument.notes.extend(bad_instrument.notes)
                    break
        # no track to merge
        else:
            if len(bad_instrument.notes) > MIN_THRES:
                good_instruments.append(bad_instrument)
    p_midi.instruments = good_instruments


def limit_max_track(p_midi, MAX_TRACK=40):
    # merge track with least notes and limit the maximum amount of track to 40
    good_instruments = p_midi.instruments
    good_instruments.sort(
        key=lambda x: (not x.is_drum, -len(x.notes)))  # place drum track or the most note track at first
    assert good_instruments[0].is_drum == True or len(good_instruments[0].notes) >= len(
        good_instruments[1].notes), tuple(len(x.notes) for x in good_instruments[:3])
    # assert good_instruments[0].is_drum == False, (, len(good_instruments[2]))
    track_idx_lst = list(range(len(good_instruments)))

    if len(good_instruments) > MAX_TRACK:
        new_good_instruments = copy.deepcopy(good_instruments[:MAX_TRACK])

        # print(midi_file_path)
        for id in track_idx_lst[MAX_TRACK:]:
            cur_ins = good_instruments[id]
            merged = False
            new_good_instruments.sort(key=lambda x: len(x.notes))
            for nid, ins in enumerate(new_good_instruments):
                if cur_ins.program == ins.program and cur_ins.is_drum == ins.is_drum:
                    new_good_instruments[nid].notes.extend(cur_ins.notes)
                    merged = True
                    break
            if not merged:
                pass
                # print('Track {:d} deprecated, program {:d}, note count {:d}'.format(
                # id, cur_ins.program, len(cur_ins.notes)))
        good_instruments = new_good_instruments
        # print(trks, probs, chosen)

    assert len(good_instruments) <= MAX_TRACK, len(good_instruments)
    for idx, good_instrument in enumerate(good_instruments):
        if good_instrument.is_drum:
            good_instruments[idx].program = 128
            good_instruments[idx].is_drum = False

    p_midi.instruments = good_instruments


def get_init_note_events(p_midi):
    # extract all notes in midi file
    note_events, note_on_ticks, note_dur_lst = [], [], []
    for track_idx, instrument in enumerate(p_midi.instruments):
        # track_idx_lst.append(track_idx)
        for note in instrument.notes:
            note_dur = note.end - note.start

            # special case: note_dur too long
            max_dur = 4 * p_midi.ticks_per_beat
            if note_dur / max_dur > 1:

                total_dur = note_dur
                start = note.start
                while total_dur != 0:
                    if total_dur > max_dur:
                        note_events.extend([[start, "ON", note.pitch, instrument.program,
                                             instrument.is_drum, track_idx, max_dur]])

                        note_on_ticks.append(start)
                        note_dur_lst.append(max_dur)

                        start += max_dur
                        total_dur -= max_dur
                    else:
                        note_events.extend([[start, "ON", note.pitch, instrument.program,
                                             instrument.is_drum, track_idx, total_dur]])
                        note_on_ticks.append(start)
                        note_dur_lst.append(total_dur)

                        total_dur = 0

            else:
                note_events.extend(
                    [[note.start, "ON", note.pitch, instrument.program, instrument.is_drum, track_idx, note_dur]])

                # for score analysis and beat estimating when score has no time signature
                note_on_ticks.append(note.start)
                note_dur_lst.append(note.end - note.start)

    note_events.sort(key=lambda x: (x[0], x[1] == "ON", x[5], x[4], x[3], x[2], x[-1]))
    note_events = list(k for k, _ in itertools.groupby(note_events))
    return note_events, note_on_ticks, note_dur_lst


def calculate_measure(p_midi, first_event_tick,
                      last_event_tick):  # calculate measures and append measure symbol to event_seq

    measure_events = []
    time_signature_changes = p_midi.time_signature_changes

    if not time_signature_changes:  # no time_signature_changes, estimate it
        raise AssertionError("No time_signature_changes")
    else:
        if time_signature_changes[0].time != 0 and \
                time_signature_changes[0].time > first_event_tick:
            raise AssertionError("First time signature start with None zero tick")

        # clean duplicate time_signature_changes
        temp_sig = []
        for idx, time_sig in enumerate(time_signature_changes):
            if idx == 0:
                temp_sig.append(time_sig)
            else:
                previous_timg_sig = time_signature_changes[idx - 1]
                if not (previous_timg_sig.numerator == time_sig.numerator
                        and previous_timg_sig.denominator == time_sig.denominator):
                    temp_sig.append(time_sig)
        time_signature_changes = temp_sig
        # print("time_signature_changes", time_signature_changes)
        for idx in range(len(time_signature_changes)):
            # calculate measures, eg: how many ticks per measure
            numerator = time_signature_changes[idx].numerator
            denominator = time_signature_changes[idx].denominator
            ticks_per_measure = p_midi.ticks_per_beat * (4 / denominator) * numerator

            cur_tick = time_signature_changes[idx].time

            if idx < len(time_signature_changes) - 1:
                next_tick = time_signature_changes[idx + 1].time
            else:
                next_tick = last_event_tick + int(ticks_per_measure)

            if ticks_per_measure.is_integer():
                for measure_start_tick in range(cur_tick, next_tick, int(ticks_per_measure)):
                    if measure_start_tick + int(ticks_per_measure) > next_tick:
                        measure_events.append([measure_start_tick, "BOM", None, None, None, None, 0])
                        measure_events.append([next_tick, "EOM", None, None, None, None, 0])
                    else:
                        measure_events.append([measure_start_tick, "BOM", None, None, None, None, 0])
                        measure_events.append(
                            [measure_start_tick + int(ticks_per_measure), "EOM", None, None, None, None, 0])
            else:
                assert False, "ticks_per_measure Error"
    return measure_events


def quantize_by_nth(nth_tick, note_events):
    # E.g., Quantize by 32nd note

    half = nth_tick / 2
    split_score = list(split_before(note_events, lambda x: x[1] == "BOM"))
    measure_durs = []
    eom_tick = 0
    for measure_id, measure in enumerate(split_score):
        bom_tick = measure[0][0]
        assert bom_tick == eom_tick, 'measure time error {bom_tick} {eom_tick}'
        eom_tick = measure[-1][0]
        mea_dur = eom_tick - bom_tick
        if mea_dur < nth_tick:  # measure duration need to be quantized
            measure_durs.append(1)
        else:
            if mea_dur % nth_tick < half:  # quantize to left
                measure_durs.append(mea_dur // nth_tick)
            else:
                measure_durs.append(mea_dur // nth_tick + 1)

        for evt in measure[1:-1]:
            assert evt[1] == 'ON', f'measure structure error {evt[1]}'
            rel_tick = evt[0] - bom_tick
            if rel_tick % nth_tick <= half:
                rel_tick = min(rel_tick // nth_tick, measure_durs[-1] - 1)
            else:
                rel_tick = min(rel_tick // nth_tick + 1, measure_durs[-1] - 1)
            evt[0] = rel_tick

    final_events = []
    lasteom = 0
    for measure_id, measure in enumerate(split_score):
        measure[0][0] = lasteom
        measure[-1][0] = measure[0][0] + measure_durs[measure_id]
        lasteom = measure[-1][0]

        for event in measure[1:-1]:
            event[0] += measure[0][0]

            if event[-1] < nth_tick:  # duration too short, quantize to 1
                event[-1] = 1
            else:
                if event[-1] % nth_tick <= half:
                    event[-1] = event[-1] // nth_tick
                else:
                    event[-1] = event[-1] // nth_tick + 1

        final_events.extend(measure)
    return final_events


def prettify(note_events, ticks_per_beat):
    fist_event_idx = next(i for i in (range(len(note_events))) if note_events[i][1] == "ON")
    last_event_idx = next(i for i in reversed(range(len(note_events))) if note_events[i][1] == "ON")

    assert note_events[fist_event_idx - 1][1] == "BOM", "measure_start Error"
    assert note_events[last_event_idx + 1][1] == "EOM", "measure_end Error"

    # remove invalid measures on both sides
    note_events = note_events[fist_event_idx - 1: last_event_idx + 2]

    # check again
    assert note_events[0][1] == "BOM", "measure_start Error"
    assert note_events[-1][1] == "EOM", "measure_end Error"

    # -------------- zero start tick -----------------
    start_tick = note_events[0][0]
    if start_tick != 0:
        for event in note_events:
            event[0] -= start_tick

    from fractions import Fraction
    ticks_32th = Fraction(ticks_per_beat, 8)

    note_events = quantize_by_nth(ticks_32th, note_events)

    note_events.sort(key=lambda x: (x[0], x[1] == "ON", x[1] == "BOM", x[1] == "EOM",
                                    x[5], x[4], x[3], x[2], x[-1]))
    note_events = list(k for k, _ in itertools.groupby(note_events))

    # -------------------------check measure duration----------------------------------------------
    note_events.sort(key=lambda x: (x[0], x[1] == "ON", x[1] == "BOM", x[1] == "EOM",
                                    x[5], x[4], x[3], x[2], x[-1]))
    split_score = list(split_before(note_events, lambda x: x[1] == "BOM"))

    check_measure_dur = [0]

    for measure_idx, measure in enumerate(split_score):
        first_tick = measure[0][0]
        last_tick = measure[-1][0]
        measure_dur = last_tick - first_tick
        if measure_dur > 100:
            raise AssertionError("Measure duration error")
        split_score[measure_idx][0][-1] = measure_dur

        if measure_dur in check_measure_dur:
            # print(measure_dur)
            raise AssertionError("Measure duration error")
    return split_score


def get_pos_and_cc(split_score):
    new_event_seq = []
    for measure_idx, measure in enumerate(split_score):
        measure.sort(key=lambda x: (x[1] == "EOM", x[1] == "ON", x[1] == 'CHR', x[1] == "BOM", x[-2]))
        bom_tick = measure[0][0]

        # split measure by track
        track_nmb = set(map(lambda x: x[-2], measure[2:-1]))
        tracks = [[y for y in measure if y[-2] == x] for x in track_nmb]

        # ---------- calculate POS for each track / add CC
        new_measure = []
        for track_idx, track in enumerate(tracks):
            pos_lst = []
            trk_abs_num = -1
            for event in track:
                if event[1] == "ON":
                    assert trk_abs_num == -1 or trk_abs_num == event[
                        -2], "Error: found inconsistent trackid within same track"
                    trk_abs_num = event[-2]
                    mypos = event[0] - bom_tick
                    pos_lst.append(mypos)
                    pos_lst = list(set(pos_lst))

            for pos in pos_lst:
                tracks[track_idx].append([pos + bom_tick, "POS", None, None, None, None, pos])
            tracks[track_idx].insert(0, [bom_tick, "CC", None, None, None, None, trk_abs_num])
            tracks[track_idx].sort(
                key=lambda x: (x[0], x[1] == "ON", x[1] == "POS", x[1] == "CC", x[5], x[4], x[3], x[2]))

        new_measure.append(measure[0])
        new_measure.append(measure[1])
        for track in tracks:
            for idx, event in enumerate(track):
                new_measure.append(event)

        new_event_seq.extend(new_measure)

    return new_event_seq


def event_seq_to_str(new_event_seq):
    char_events = []

    for evt in new_event_seq:
        if evt[1] == 'ON':
            char_events.append(pit2str(evt[2]))  # pitch
            char_events.append(dur2str(evt[-1]))  # duration
            char_events.append(trk2str(evt[-2]))  # track
            char_events.append(ins2str(evt[3]))  # instrument
        elif evt[1] == 'POS':
            char_events.append(pos2str(evt[-1]))  # type (time position)
            char_events.append('RZ')
            char_events.append('TZ')
            char_events.append('YZ')
        elif evt[1] == 'BOM':
            char_events.append(bom2str(evt[-1]))
            char_events.append('RZ')
            char_events.append('TZ')
            char_events.append('YZ')
        elif evt[1] == 'CC':
            char_events.append('NT')
            char_events.append('RZ')
            char_events.append('TZ')
            char_events.append('YZ')
        elif evt[1] == 'CHR':
            char_events.append('H' + evt[-1])
            char_events.append('RZ')
            char_events.append('TZ')
            char_events.append('YZ')
        else:
            assert False, ("evt type error", evt[1])
    return char_events


# abs_pos type pitch program is_drum track_id duration/rela_pos
def midi_to_event_seq_str(midi_file_path, readonly=False):
    p_midi = MidiFile(midi_file_path)
    for ins in p_midi.instruments:
        ins.remove_invalid_notes(verbose=False)

    merge_drums(p_midi)

    if not readonly:
        merge_sparse_track(p_midi)

    limit_max_track(p_midi)

    note_events, note_on_ticks, _ = get_init_note_events(p_midi)

    measure_events = calculate_measure(p_midi, min(note_on_ticks), max(note_on_ticks))
    note_events.extend(measure_events)
    note_events.sort(key=lambda x: (x[0], x[1] == "ON", x[1] == "BOM", x[1] == "EOM",
                                    x[5], x[4], x[3], x[2]))

    split_score = prettify(note_events, p_midi.ticks_per_beat)

    for measure_idx, measure in enumerate(split_score):  # calculate chord for every measure
        chord_evt = measure_calc_chord(measure)
        split_score[measure_idx].insert(1, chord_evt)

    new_event_seq = get_pos_and_cc(split_score)

    char_events = event_seq_to_str(new_event_seq)

    return char_events


def mp_worker(file_path):
    try:
        event_seq = midi_to_event_seq_str(file_path)
        return event_seq
    except (OSError, EOFError, ValueError, KeyError) as e:
        print(file_path)
        traceback.print_exc(limit=0)
        print()
        return "error"

    except AssertionError as e:
        if str(e) == "No time_signature_changes":
            return "error"
        elif str(e) == "Measure duration error":
            # print("Measure duration error", file_path)
            return "error"
        else:
            print("Other Assertion Error", str(e), file_path)
            return "error"

    except Exception as e:
        print(file_path)
        traceback.print_exc(limit=0)
        print()
        return "error"


def mp_handler(file_paths):
    start = time.time()

    broken_counter = 0
    good_counter = 0

    event_seq_res = []
    chord_cnter = Counter()
    print(f'Processing {len(file_paths)} midis with {WORKERS} processes')

    with multiprocessing.Pool(WORKERS) as p:
        for event_seq in p.imap(mp_worker, file_paths):
            if isinstance(event_seq, str):
                broken_counter += 1
            elif len(event_seq) > 0:
                event_seq_res.append(event_seq)
                good_counter += 1
            else:
                broken_counter += 1

    print(f"MIDI data preprocessing takes: {time.time() - start}s, "
          f"{good_counter} samples collected, {broken_counter} broken.")

    # ----------------------------------------------------------------------------------
    txt_start = time.time()
    if not os.path.exists('Data/Glob/Preprocessed/'):
        os.makedirs('Data/Glob/Preprocessed/')

    with open("Data/Glob/Preprocessed/raw_corpus.txt", "w", encoding="utf-8") as f:
        for idx, piece in enumerate(event_seq_res):
            f.write(' '.join(piece) + '\n')

    print("Create txt file takes: ", time.time() - txt_start)

# endregion Preprocess_MIDI


# region Get_BPE_Data
RATIO = 4
MERGE_CNT = 700
CHAR_CNT = 128


def resort(voc: str) -> str:
    assert (len(voc) % 2 == 0), voc
    pitch_set = list(set(voc[i:i + 2] for i in range(0, len(voc), 2)))
    assert len(pitch_set) * 2 == len(voc), voc
    return ''.join(sorted(pitch_set, key=str2pit))


def gettokens(voc: set, merges):
    assert len(voc) > 1, voc
    last_idx = 0
    while (len(voc) > 1):
        flag = False
        for i in range(last_idx, len(merges)):
            t1, t2, t3 = merges[i]
            if t1 in voc and t2 in voc:
                voc.remove(t1)
                voc.remove(t2)
                voc.add(t3)
                flag = True
                last_idx = i + 1
                break
        if not flag:
            break
    return voc


def merge_mulpies(new_toks, mulpies, other, merges, merged_vocs, divide_res):
    assert other is not None, mulpies
    for dur, mulpi in mulpies.items():
        if len(mulpi) > 1:  # apply bpe (with saved tokenization method)
            mulpi_sorted = tuple(sorted(list(mulpi), key=str2pit))
            if mulpi_sorted in divide_res:
                submulpies = divide_res[mulpi_sorted]
            else:
                submulpies = sorted(gettokens(set(str2pit(x) for x in mulpi_sorted), merges))

            for submulpi_num in submulpies:
                new_toks.extend([merged_vocs[submulpi_num], dur] + other)
        else:
            new_toks.extend([list(mulpi)[0], dur] + other)


def apply_bpe_for_sentence(toks, merges, merged_vocs, divide_res, ratio=RATIO):
    if isinstance(toks, str):
        toks = toks.split()
    new_toks = []
    mulpies = dict()
    other = None

    for idx in range(0, len(toks), ratio):
        e, d = toks[idx:idx + 2]
        if not ispitch(e):
            if len(mulpies) > 0:
                merge_mulpies(new_toks, mulpies, other, merges, merged_vocs, divide_res)
                mulpies = dict()
            new_toks.extend(toks[idx:idx + ratio])
        else:
            mulpies.setdefault(d, set()).add(e)
            other = toks[idx + 2:idx + ratio]

    if len(mulpies) > 0:
        merge_mulpies(new_toks, mulpies, other, merges, merged_vocs, divide_res)

    assert len(new_toks) % ratio == 0, f'error new token len {len(new_toks)}'

    return new_toks


def load_before_apply_bpe(bpe_res_dir):
    merged_vocs = [pit2str(i) for i in range(CHAR_CNT)]
    merged_voc_to_int = {pit2str(i): i for i in range(CHAR_CNT)}
    merges = []
    with open(bpe_res_dir + 'codes.txt', 'r') as f:
        for line in f:
            a, b, _ = line.strip().split()
            a, b, ab = resort(a), resort(b), resort(a + b)

            a_ind, b_ind, ab_ind = merged_voc_to_int[a], merged_voc_to_int[b], len(merged_vocs)
            merges.append((a_ind, b_ind, ab_ind))

            merged_voc_to_int[ab] = ab_ind
            merged_vocs.append(ab)

    return merges, merged_vocs


def apply_bpe_for_word_dict(mulpi_list, merges):
    # apply bpe for vocabs
    bpe_freq = Counter()
    divided_bpe_total = Counter()
    divide_res = dict()
    for ori_voc, cnt in tqdm(mulpi_list):
        ret = sorted(gettokens(set(str2pit(x) for x in ori_voc), merges))
        divide_res[ori_voc] = ret
        divided_bpe_total[len(ret)] += cnt
        for r in ret:
            bpe_freq[merged_vocs[r]] += cnt

    return divide_res, divided_bpe_total, bpe_freq


def count_single_mulpies(toks, ratio=RATIO):
    if isinstance(toks, str):
        toks = toks.split()
    mulpies = dict()
    chord_dict = Counter()
    l_toks = len(toks)
    for idx in range(0, l_toks, ratio):
        e, d = toks[idx:idx + 2]

        if not ispitch(e):
            if len(mulpies) > 0:
                for dur, mulpi in mulpies.items():
                    if len(mulpi) > 1:
                        chord_dict[tuple(sorted(list(mulpi), key=str2pit))] += 1
                mulpies = dict()
        else:
            mulpies.setdefault(d, set()).add(e)

    if len(mulpies) > 0:
        for dur, mulpi in mulpies.items():
            if len(mulpi) > 1:
                chord_dict[tuple(sorted(list(mulpi), key=str2pit))] += 1

    return chord_dict, l_toks // ratio

# endregion Get_BPE_Data


# region Make_Data
PAD = 1
EOS = 2
BOS = 0

# RATIO = 4
# SAMPLE_LEN_MAX = 4096
# SOR = 4


# def get_mea_cnt(str_toks, ratio):
#     bom_idx = []
#     for idx in range(0, len(str_toks), ratio):
#         if str_toks[idx][0].lower() == 'm':
#             bom_idx.append(idx) # extract all bom tokens idx
#     bom_idx.append(len(str_toks))
#     ret = 0
#     for id, nid in zip(bom_idx[:-1], bom_idx[1:]):
#         ret += 1
#     return ret

def process_single_piece(bundle_input, ratio, sample_len_max):
    line, str2int = bundle_input

    if isinstance(line, str):
        str_toks = line.split()
    else:
        str_toks = line

    measures = []
    cur_mea = []
    max_rel_pos = 0
    mea_tok_lengths = []

    rel_pos = 0

    for idx in range(0, len(str_toks), ratio):
        c = str_toks[idx][0]

        if c.lower() == 'm':  # BOM Token
            if len(cur_mea) > 0:  # exlude first bom
                measures.append(cur_mea)
                mea_tok_lengths.append(len(cur_mea) // (ratio + 1))
            cur_mea = []
            if rel_pos > max_rel_pos:
                max_rel_pos = rel_pos
            rel_pos = 0
        elif c.lower() == 'h':  # chord token
            if rel_pos > max_rel_pos:
                max_rel_pos = rel_pos
            rel_pos = 0
        elif c.lower() == 'n':  # CC/NT Token
            if rel_pos > max_rel_pos:
                max_rel_pos = rel_pos
            rel_pos = 1
        elif c.lower() == 'p':  # pos token
            rel_pos += 2
        else:  # on token
            pass
        cur_mea += [str2int[x] for x in str_toks[idx:idx + ratio]] + [rel_pos - 1 if c.lower() == 'p' else rel_pos]
    # TODO: how to design rel_pos and measure pos?
    if len(cur_mea) > 0:
        measures.append(cur_mea)
        mea_tok_lengths.append(len(cur_mea) // (ratio + 1))
        if rel_pos > max_rel_pos:
            max_rel_pos = rel_pos

    # tmp = get_mea_cnt(str_toks, ratio)
    # if get_mea_cnt(str_toks, ratio) != len(measures):
    #     print(f'{tmp} {len(measures)} {len(mea_tok_lengths)}')

    len_cnter = Counter()
    for l in mea_tok_lengths:
        len_cnter[l // 10] += 1

    for idx in range(1, len(mea_tok_lengths)):
        mea_tok_lengths[idx] += mea_tok_lengths[idx - 1]

    def get_cur_tokens(s, t):  # return total cnt of tokens in measure [s, t]
        return mea_tok_lengths[t] - (mea_tok_lengths[s - 1] if s > 0 else 0)

    maxl = 1
    for s in range(len(mea_tok_lengths)):
        t = s + maxl - 1

        while t < len(mea_tok_lengths) and get_cur_tokens(s, t) < sample_len_max:
            t += 1

        t = min(t, len(mea_tok_lengths) - 1)
        maxl = max(maxl, t - s + 1)

    return measures, len_cnter, max_rel_pos, maxl


def myshuffle(l):
    ret = []
    idx = list(range(len(l)))
    random.shuffle(idx)
    for id in idx:
        ret.append(l[id])
    return ret


def mp_handler2(raw_data, str2int, output_file, ratio, sample_len_max, num_workers=WORKERS):
    begin_time = time.time()

    merged_sentences = []
    mea_cnt_dis = Counter()
    mea_len_dis = Counter()
    max_rel_pos = 0
    maxl = 0
    with multiprocessing.Pool(num_workers) as p:
        for sentences, len_cnter, pos, l in p.imap_unordered(
                partial(process_single_piece, ratio=ratio, sample_len_max=sample_len_max),
                [(x, str2int) for x in raw_data]):
            merged_sentences.append(sentences)
            mea_len_dis += len_cnter
            max_rel_pos = max(max_rel_pos, pos)
            maxl = max(maxl, l)
    for sentences in merged_sentences:
        mea_cnt_dis[len(sentences) // 10] += 1

    print(f'measure collection finished, total {sum(len(x) for x in merged_sentences)} measures, '
          f'time elapsed: {time.time() - begin_time} s')
    print(f'max cnt in a sample (rel_pos, measure): {max_rel_pos}, {maxl}')

    begin_time = time.time()

    if output_file.split('/')[-1] == 'train':
        with open('vocab.sh', 'a') as f:
            f.write(f'MAX_REL_POS={max_rel_pos + 5}\n')
            f.write(f'MAX_MEA_POS={maxl * 3 + 5}\n')
        with open('Data/Glob/Preprocessed/mea_cnt_dis.txt', 'w') as f:
            for k, v in sorted(mea_cnt_dis.items()):
                f.write(f'{k * 10} {v}\n')
        with open('Data/Glob/Preprocessed/mea_len_dis.txt', 'w') as f:
            for k, v in sorted(mea_len_dis.items()):
                f.write(f'{k * 10} {v}\n')

    # TODO: modify to not use fairseq; use tensorflow instead
    # ds = MMapIndexedDatasetBuilder(output_file + '.bin', dtype=np.uint16)
    # for doc in tqdm(merged_sentences, desc='writing bin file'):
    #     for sentence in doc:
    #         ds.add_item(torch.IntTensor(sentence))
    #     ds.add_item(torch.IntTensor([EOS]))
    # ds.finalize(output_file + '.idx')
    # Convert the data to tf.train.Example and write to TFRecord
    def serialize_example(sequence):
        """Converts a sequence into a tf.train.Example"""
        feature = {
            'sequence': tf.train.Feature(int64_list=tf.train.Int64List(value=sequence))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    with tf.io.TFRecordWriter(output_file + '.tfrecord') as writer:
        for doc in tqdm(merged_sentences, desc='writing bin file'):
            for sentence in doc:
                example = serialize_example(sentence)
                writer.write(example)
            writer.write(serialize_example([EOS]))

    print(f'Bin file writing finished, time elapsed: {time.time() - begin_time} s')


def makevocabs(line, ratio):
    toks = line.split()
    ret_sets = []
    for i in range(ratio):
        sub_toks = toks[i::ratio]
        ret_sets.append(set(sub_toks))
    return ret_sets

# endregion Make_Data


if __name__ == '__main__':
    # region preprocess_midi
    warnings.filterwarnings('ignore')

    folder_path = "Data/MIDI/VoiceParts/Combined"
    file_paths = []
    for path, directories, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".mid") or file.endswith(".MID"):
                file_path = path + "/" + file
                file_paths.append(file_path)

    # run multi-processing midi extractor
    mp_handler(file_paths)
    # endregion preprocess_midi

    # region get_bpe_data
    start_time = time.time()

    paragraphs = []

    raw_data_path = 'Data/Glob/Preprocessed/raw_corpus.txt'
    merged_data_path = 'Data/Glob/Preprocessed/raw_corpus_bpe.txt'
    output_dir = 'Data/Glob/Preprocessed/bpe_res/'
    os.makedirs(output_dir, exist_ok=True)
    raw_data = []
    with open(raw_data_path, 'r') as f:
        for line in tqdm(f, desc="reading original txt file..."):
            raw_data.append(line.strip())

    chord_dict = Counter()
    before_total_tokens = 0
    for sub_chord_dict, l_toks in p_uimap(count_single_mulpies, raw_data, num_cpus=WORKERS):
        chord_dict += sub_chord_dict
        before_total_tokens += l_toks

    mulpi_list = sorted(chord_dict.most_common(), key=lambda x: (-x[1], x[0]))
    with open(output_dir + 'ori_voc_cnt.txt', 'w') as f:
        f.write(str(len(mulpi_list)) + '\n')
        for k, v in mulpi_list:
            f.write(''.join(k) + ' ' + str(v) + '\n')
    with open(output_dir + 'codes.txt', 'w') as stdout:
        with open(output_dir + 'merged_voc_list.txt', 'w') as stderr:
            subprocess.run(['./music_bpe_exec', 'learnbpe', f'{MERGE_CNT}', output_dir + 'ori_voc_cnt.txt'],
                           stdout=stdout, stderr=stderr)
    print(f'learnBPE finished, time elapsed:　{time.time() - start_time}')
    start_time = time.time()

    merges, merged_vocs = load_before_apply_bpe(output_dir)
    divide_res, divided_bpe_total, bpe_freq = apply_bpe_for_word_dict(mulpi_list, merges)
    with open(output_dir + 'divide_res.json', 'w') as f:
        json.dump({' '.join(k): v for k, v in divide_res.items()}, f)
    with open(output_dir + 'bpe_voc_cnt.txt', 'w') as f:
        for voc, cnt in bpe_freq.most_common():
            f.write(voc + ' ' + str(cnt) + '\n')
    ave_len_bpe = sum(k * v for k, v in divided_bpe_total.items()) / sum(divided_bpe_total.values())
    ave_len_ori = sum(len(k) * v for k, v in mulpi_list) / sum(v for k, v in mulpi_list)
    print(f'average mulpi length original:　{ave_len_ori}, average mulpi length after bpe: {ave_len_bpe}')
    print(f'applyBPE for word finished, time elapsed:　{time.time() - start_time}')
    start_time = time.time()

    # applyBPE for corpus

    after_total_tokens = 0
    with open(merged_data_path, 'w') as f:
        for x in tqdm(raw_data, desc="writing bpe data"):  # unable to parallelize for out of memory
            new_toks = apply_bpe_for_sentence(x, merges, merged_vocs, divide_res)
            after_total_tokens += len(new_toks) // RATIO
            f.write(' '.join(new_toks) + '\n')
    print(f'applyBPE for corpus finished, time elapsed:　{time.time() - start_time}')
    print(
        f'before tokens: {before_total_tokens}, after tokens: {after_total_tokens}, delta: {(before_total_tokens - after_total_tokens) / before_total_tokens}')
    # endregion get_bpe_data

    # region make_data
    SEED, SAMPLE_LEN_MAX, totpiece, RATIO, bpe, map_meta_to_pad = None, None, None, None, None, None
    print('config.sh: ')
    with open('config.sh', 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                break
            print(line)
            line = line.split('=')
            assert len(line) == 2, f'invalid config {line}'
            if line[0] == 'SEED':
                SEED = int(line[1])
                random.seed(SEED)
            elif line[0] == 'MAX_POS_LEN':
                SAMPLE_LEN_MAX = int(line[1])
            elif line[0] == 'MAXPIECES':
                totpiece = int(line[1])
            elif line[0] == 'RATIO':
                RATIO = int(line[1])
            elif line[0] == 'BPE':
                bpe = int(line[1])
            elif line[0] == 'IGNORE_META_LOSS':
                map_meta_to_pad = int(line[1])

    assert SEED is not None, "missing arg: SEED"
    assert SAMPLE_LEN_MAX is not None, "missing arg: MAX_POS_LEN"
    assert totpiece is not None, "missing arg: MAXPIECES"
    assert RATIO is not None, "missing arg: RATIO"
    assert bpe is not None, "missing arg: BPE"
    assert map_meta_to_pad is not None, "missing arg: IGNORE_META_LOSS"

    bpe = "" if bpe == 0 else "_bpe"
    raw_corpus = f'raw_corpus{bpe}'
    model_name = f"linear_{SAMPLE_LEN_MAX}_chord{bpe}"
    raw_data_path = f'Data/Glob/Preprocessed/{raw_corpus}.txt'
    output_dir = f'Data/Glob/Preprocessed/Model_spec/{model_name}_hardloss{map_meta_to_pad}/'

    start_time = time.time()
    raw_data = []
    with open(raw_data_path, 'r') as f:
        for line in tqdm(f, desc='reading...'):
            raw_data.append(line.strip())
            if len(raw_data) >= totpiece:
                break

    sub_vocabs = dict()
    for i in range(RATIO):
        sub_vocabs[i] = set()

    for ret_sets in p_uimap(partial(makevocabs, ratio=RATIO), raw_data, num_cpus=WORKERS, desc='setting up vocabs'):
        for i in range(RATIO):
            sub_vocabs[i] |= ret_sets[i]

    voc_to_int = dict()
    for type in range(RATIO):
        sub_vocabs[type] |= set(('<bos>', '<pad>', '<eos>', '<unk>'))
        sub_vocabs[type] -= set(('RZ', 'TZ', 'YZ'))
        sub_vocabs[type] = sorted(list(sub_vocabs[type]), key=sort_tok_str)
        voc_to_int.update({v: i for i, v in enumerate(sub_vocabs[type])})
    output_dict = sorted(list(set(voc_to_int.values())))
    max_voc_size = max(output_dict)
    print("max voc idx: ", max_voc_size)

    os.makedirs(output_dir + 'bin/', exist_ok=True)
    with open(output_dir + 'bin/dict.txt', 'w') as f:
        for i in range(4, max_voc_size + 1):  # [4, max_voc_size]
            f.write("%d 0\n" % i)

    os.makedirs(output_dir + 'vocabs/', exist_ok=True)
    for type in range(RATIO):
        sub_vocab = sub_vocabs[type]
        with open(output_dir + 'vocabs/vocab_%d.json' % type, 'w') as f:
            json.dump({i: v for i, v in enumerate(sub_vocab)}, f)
    with open(output_dir + 'vocabs/ori_dict.json', 'w') as f:
        json.dump(voc_to_int, f)
    print('sub vocab size:', end=' ')
    for type in range(RATIO):
        print(len(sub_vocabs[type]), end=' ')
    print()
    with open(f'vocab.sh', 'w') as f:
        for type in range(RATIO):
            f.write(f'SIZE_{type}={len(sub_vocabs[type])}\n')

    totpiece = len(raw_data)
    print("total pieces: {:d}, create dict time: {:.2f} s".format(totpiece, time.time() - start_time))

    raw_data = myshuffle(raw_data)
    os.makedirs(output_dir + 'bin/', exist_ok=True)
    train_size = min(int(totpiece * 0.99), totpiece - 2)
    splits = {'train': raw_data[:train_size], 'valid': raw_data[train_size:-1], 'test': raw_data[-1:]}

    voc_to_int.update({x: (PAD if map_meta_to_pad == 1 else BOS) for x in ('RZ', 'TZ', 'YZ')})
    for mode in splits:
        print(mode)
        mp_handler2(splits[mode], voc_to_int, output_dir + f'bin/{mode}', ratio=RATIO, sample_len_max=SAMPLE_LEN_MAX)
    # endregion make_data
