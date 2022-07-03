import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

LABEL_MAPPING = {
    0: 'unknown',
    1: 'neutral',
    2: 'positive',
    3: 'negative',
}


@dataclass
class PSMMSample:
    text: str
    title: str
    clip_id: str
    project_id: str
    scrape_date: Optional[datetime] = None
    url: Optional[str] = None
    base_url: Optional[str] = None
    label: Optional[str] = None
    label_date: Optional[datetime] = None


_HOUR = r'\d{1,2}:\d{1,2}'
DATE_TIME_REGEXES = (
    (fr'\s(\d\d-\d\d-\d\d\d\d\s{_HOUR})$', '%d-%m-%Y %H:%M'),
    (fr'\s(\d\d-\d\d-\d\d\s{_HOUR})$', '%d-%m-%y %H:%M'),
    (fr'\s(\d\d\d\d-\d\d-\d\d\s{_HOUR})$', '%Y-%m-%d %H:%M'),
)
URL_REGEX = re.compile(r'^.+?[^/:](?=[?/]|$)')


def load_psmm_data(
        path: Path,
) -> Iterable[PSMMSample]:
    clips_path = path / 'clips'
    projects_path = path / 'projects'

    clip_paths = [
        clip_path for parent_dir in clips_path.iterdir() for clip_path in parent_dir.iterdir()
    ]

    progress_bar = tqdm(
        total=len(clip_paths),
        desc='Creating a project-to-clip mapping',
    )
    project_mapping = _create_project_mapping(projects_path)

    progress_bar.set_description('Loading clips')
    for clip_path in clip_paths:
        clip_id = clip_path.stem
        for project_label in project_mapping.get(clip_id, []):
            yield _create_pssmm_example(
                clip_path,
                clip_id=clip_id,
                label_id=project_label['label_id'],
                project_id=project_label['project_id'],
                label_date_str=project_label['label_date_str'],
            )
        progress_bar.update()


def _create_project_mapping(
        projects_path: Path,
) -> Dict[str, List[Dict[str, str]]]:
    mapping = defaultdict(list)
    for project in projects_path.iterdir():
        for labels_filename in project.iterdir():
            with projects_path.joinpath(project, labels_filename).open('r') as label_file:
                reader = csv.reader(label_file, delimiter='\t')
                for clip_id, label_id, date_str in reader:
                    mapping[clip_id].append(
                        {
                            'label_id': label_id,
                            'project_id': project.stem,
                            'label_date_str': date_str,
                        }
                    )
    return mapping


def _create_pssmm_example(
        sample_path: Path,
        clip_id: str,
        label_id: str,
        project_id: str,
        label_date_str: str,
) -> PSMMSample:
    lines = sample_path.read_text().splitlines()
    title, _, url_str, _, *text_lines = lines
    label_date = datetime.strptime(label_date_str, '%Y-%m-%d %H:%M:%S')

    scrape_date = None
    for regex, pattern in DATE_TIME_REGEXES:
        date_match = re.search(regex, title)
        if date_match:
            title = title[: date_match.start()][:-3]  # two spaces and one '-'
            group = date_match.group(1)
            scrape_date = datetime.strptime(group, pattern)
            assert scrape_date is not None
            break

    base_url = None
    url = None
    if url_str:
        url = url_str
        url_match = re.match(URL_REGEX, url_str)
        if url_match:
            base_url = url_match.group()

    text = '\n'.join(text_lines).strip()

    return PSMMSample(
        text=text,
        title=title.strip(),
        url=url,
        base_url=base_url,
        clip_id=clip_id,
        project_id=project_id,
        scrape_date=scrape_date,
        label=LABEL_MAPPING[int(label_id)],
        label_date=label_date,
    )
