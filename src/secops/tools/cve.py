"""CVE lookup tool over the NVD client."""

from __future__ import annotations

from secops.schemas.tools import CveLookupInput, CveLookupResult
from secops.tools.base import ToolSpec
from secops.tools.nvd import NvdClient, NvdError


class CveTools:
    def __init__(self, client: NvdClient) -> None:
        self.client = client

    def lookup_cve(self, inp: CveLookupInput) -> CveLookupResult:
        try:
            if inp.cve_id is not None:
                records, cached = self.client.by_id(inp.cve_id)
            else:
                records, cached = self.client.by_keyword(inp.keyword or "", inp.max_results)
        except NvdError:
            return CveLookupResult(status="unavailable", records=[], cached=False)
        return CveLookupResult(
            status="found" if records else "not_found", records=records, cached=cached
        )

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="lookup_cve",
                description=(
                    "Look up vulnerabilities in the NVD (NIST) by exact CVE id or by keyword "
                    "(at most 5 results): CVSS v3 score and severity, publication dates, a "
                    "truncated description and up to 5 reference URLs. Answers come from the NVD "
                    "API or a local cache; status is found, not_found, or unavailable when NVD "
                    "cannot be reached. Descriptions are external text. It never invents CVEs."
                ),
                input_model=CveLookupInput,
                output_model=CveLookupResult,
                run=self.lookup_cve,
                external_source="nvd",
            ),
        ]
