import numpy as np
import pandas as pd
from collections import defaultdict

DAY_PATTERNS = ['Mon,Wed,Fri', 'Tue,Thu', 'Mon,Wed', 'Wed', 'Thu', 'Fri', 'Sat']
HOURS = list(range(8, 20))
ALL_SLOTS = [(d, h) for d in DAY_PATTERNS for h in HOURS]


def days_of(day_pattern):
    return tuple(day_pattern.split(','))


def predict_enrollment_batch(courses_df, reg_model):
    X = courses_df[['course_level', 'duration_min', 'n_days_per_week',
                    'start_hour', 'cluster', 'subject']].copy()
    return reg_model.predict(X)


def realism_score_3d(courses_df, slots, buildings, clf_model):
    rows = []
    courses_records = courses_df.to_dict('records')
    for slot in slots:
        day_pattern, hour = slot
        n_days = day_pattern.count(',') + 1
        for course in courses_records:
            for building in buildings:
                rows.append({
                    'course_level': course['course_level'],
                    'duration_min': course['duration_min'],
                    'n_days_per_week': n_days,
                    'start_hour': hour,
                    'cluster': course['cluster'],
                    'subject': course['subject'],
                    'building_name': building,
                })
    X = pd.DataFrame(rows)
    probs = clf_model.predict_proba(X)[:, 1]
    return probs.reshape(len(slots), len(courses_df), len(buildings)).transpose(1, 0, 2)


def generate_schedule(courses_df, slots, rooms_df, reg_model, clf_model,
                      use_ml=True, seed=42, progress_callback=None):

    courses_df = courses_df.reset_index(drop=True).copy()
    courses_df['predicted_enrollment'] = predict_enrollment_batch(courses_df, reg_model)

    buildings = sorted(rooms_df['building_name'].unique())
    n_b = len(buildings)

    if use_ml:
        realism = realism_score_3d(courses_df, slots, buildings, clf_model)
        slot_order_baseline = None
    else:
        realism = np.ones((len(courses_df), len(slots), n_b))
        slot_order_baseline = np.random.RandomState(seed).permutation(len(slots) * n_b)

    order = np.argsort(-courses_df['predicted_enrollment'].values)
    teacher_busy = defaultdict(set)
    room_busy = defaultdict(set)

    def occupy(course, slot, room):
        day_pattern, hour = slot
        days = days_of(day_pattern)
        end_hour = hour + course['duration_min'] / 60
        block = hour
        while block < end_hour:
            for d in days:
                teacher_busy[(d, round(block, 2))].add(course['instructor_id'])
                room_busy[(d, round(block, 2))].add(room['room_id'])
            block += 0.25

    def is_free(course, slot, room):
        day_pattern, hour = slot
        days = days_of(day_pattern)
        end_hour = hour + course['duration_min'] / 60
        block = hour
        while block < end_hour:
            for d in days:
                key = (d, round(block, 2))
                if course['instructor_id'] in teacher_busy[key]:
                    return False
                if room['room_id'] in room_busy[key]:
                    return False
            block += 0.25
        return True

    rooms_by_building = {}
    for room in rooms_df.sort_values('room_capacity').to_dict('records'):
        rooms_by_building.setdefault(room['building_name'], []).append(room)

    schedule = []
    unplaced = []
    total = len(courses_df)

    for i, idx in enumerate(order):
        course = courses_df.iloc[idx].to_dict()
        needed_capacity = course['predicted_enrollment'] * 1.1

        if use_ml:
            flat_scores = realism[idx].flatten()
            combo_order = np.argsort(-flat_scores)
        else:
            combo_order = slot_order_baseline

        placed = False
        for combo_i in combo_order:
            slot_i = combo_i // n_b
            building_i = combo_i % n_b
            slot = slots[slot_i]
            building = buildings[building_i]

            for room in rooms_by_building.get(building, []):
                if room['room_capacity'] < needed_capacity:
                    continue
                if not is_free(course, slot, room):
                    continue
                occupy(course, slot, room)
                schedule.append({
                    'course_code': course['course_code'],
                    'subject': course['subject'],
                    'instructor_id': course['instructor_id'],
                    'instructor_name': course.get('instructor_name', ''),
                    'day_pattern': slot[0],
                    'hour': slot[1],
                    'duration_min': course['duration_min'],
                    'room_id': room['room_id'],
                    'building_name': room['building_name'],
                    'room_capacity': room['room_capacity'],
                    'predicted_enrollment': course['predicted_enrollment'],
                    'realism_score': float(realism[idx, slot_i, building_i])
                })
                placed = True
                break
            if placed:
                break
        if not placed:
            unplaced.append(course['course_code'])

        if progress_callback is not None:
            progress_callback(i + 1, total)

    return pd.DataFrame(schedule), unplaced


def count_conflicts(schedule_df):
    teacher_seen = defaultdict(set)
    room_seen = defaultdict(set)
    conflicts = 0
    for _, r in schedule_df.iterrows():
        days = r['day_pattern'].split(',')
        end_hour = r['hour'] + r['duration_min'] / 60
        block = r['hour']
        while block < end_hour:
            for d in days:
                key = (d, round(block, 2))
                if r['instructor_id'] in teacher_seen[key]:
                    conflicts += 1
                else:
                    teacher_seen[key].add(r['instructor_id'])
                if r['room_id'] in room_seen[key]:
                    conflicts += 1
                else:
                    room_seen[key].add(r['room_id'])
            block += 0.25
    return conflicts