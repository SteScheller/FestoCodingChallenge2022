from dataclasses import dataclass
from typing import List


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


if __name__ == "__main__":
    entries = read_office_database("./office_database.txt")
    print(entries)
