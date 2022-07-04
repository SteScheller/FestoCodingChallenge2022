from typing import List, Tuple
from functools import reduce
from dataclasses import dataclass


@dataclass
class OfficeDatabaseEntry:
    "Class for office_database.txt entries"
    username: str
    user_id: int
    access_key: int
    first_login_time: str

    def __hash__(self):
        return hash(
            self.username + str(self.user_id) + str(self.access_key) + self.first_login_time
        )


def read_office_database(file_path: str) -> List[OfficeDatabaseEntry]:
    with open(file_path) as f:
        content = f.readlines()

    entries = list()
    for i, line in enumerate(content):
        try:
            name, uid, key, ltime = line.split(",")
            name = name.strip()
            uid = int(uid)
            key = int(key)
            ltime = ltime.strip()
            entries.append(
                OfficeDatabaseEntry(
                    username=name, user_id=uid, access_key=key, first_login_time=ltime
                )
            )
        except Exception:
            print(f"failed to parse database line {i}: {line}")

    return entries


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
        if ((hour == max_hour) and (minute < max_minute)) or (hour < max_hour):
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
