from typing import List, Tuple
from functools import reduce

from parsing import read_office_database, OfficeDatabaseEntry


def filter_id(entries: List[OfficeDatabaseEntry], sub_id: str) -> List[OfficeDatabaseEntry]:
    filtered = list()
    for item in entries:
        if sub_id in str(item.user_id):
            filtered.append(item)

    return filtered


def filter_key(entries: List[OfficeDatabaseEntry], access_mask: int) -> List[OfficeDatabaseEntry]:
    filtered = list()
    for item in entries:
        if bool(access_mask & item.access_key):
            filtered.append(item)

    return filtered


def filter_time(entries: List[OfficeDatabaseEntry], max_time: str) -> List[OfficeDatabaseEntry]:
    def get_hour_minute(time: str) -> Tuple[int, int]:
        hour, minute = time.split(":")
        return int(hour), int(minute)

    max_hour, max_minute = get_hour_minute(max_time)

    filtered = list()
    for item in entries:
        hour, minute = get_hour_minute(item.first_login_time)
        if (hour <= max_hour) and (minute < max_minute):
            filtered.append(item)

    return filtered


if __name__ == "__main__":
    entries = read_office_database("./office_database.txt")

    filtered_suspects = list()

    # puzzle 1
    filtered = filter_id(entries, "814")
    sum_ids = reduce(lambda x, y: x + y.user_id, filtered, 0)
    print(f"Solution for puzzle 1: {sum_ids}")
    filtered_suspects.append(filtered)

    # puzzle 2
    filtered = filter_key(entries, 0b00001000)
    sum_ids = reduce(lambda x, y: x + y.user_id, filtered, 0)
    print(f"Solution for puzzle 2: {sum_ids}")
    filtered_suspects.append(filtered)

    # puzzle 3
    filtered = filter_time(entries, "7:14")
    sum_ids = reduce(lambda x, y: x + y.user_id, filtered, 0)
    print(f"Solution for puzzle 3: {sum_ids}")
    filtered_suspects.append(filtered)

    # list of suspects
    suspects = set(filtered_suspects[0])
    for suspect_list in filtered_suspects:
        suspects = suspects & set(suspect_list)
    print(f"Remaining set of suspects: {suspects}")
