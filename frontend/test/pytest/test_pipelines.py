# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for pipeline options and utility functions.
"""

import pytest

from catalyst.pipelines import insert_after_pass, insert_before_pass


@pytest.fixture(name="pipeline", scope="function")
def fixture_pipeline():
    """Fixture that returns a sample pass pipeline with a list of made-up pass names"""
    return ["pass1", "pass2", "pass3"]


class TestPassInsertion:
    """Test insertion of a pass into an existing pass pipeline."""

    # Tests for `insert_after_pass`
    def test_insert_after_pass(self, pipeline):
        """Test the ``insert_after_pass`` function with valid inputs, inserting into the middle of
        the pipeline.
        """
        new_pass_1 = "new_pass_1"
        insert_after_pass(pipeline, ref_pass="pass1", new_pass=new_pass_1)

        pipeline_expected = ["pass1", new_pass_1, "pass2", "pass3"]
        assert pipeline == pipeline_expected

    def test_insert_after_pass_at_end(self, pipeline):
        """Test the ``insert_after_pass`` function with valid inputs, inserting at the end of the
        pipeline.
        """
        new_pass_1 = "new_pass_1"
        insert_after_pass(pipeline, ref_pass="pass3", new_pass=new_pass_1)

        pipeline_expected = ["pass1", "pass2", "pass3", new_pass_1]
        assert pipeline == pipeline_expected

    @pytest.mark.parametrize("ref_pass", ["not_a_pass", 1, [], None])
    def test_insert_after_invalid_ref_pass(self, pipeline, ref_pass):
        """Test the ``insert_after_pass`` function with an invalid reference pass name."""
        new_pass = "new_pass_1"

        with pytest.raises(ValueError, match=f"Cannot insert pass '{new_pass}' into pipeline"):
            insert_after_pass(pipeline, ref_pass=ref_pass, new_pass=new_pass)

    # Tests for `insert_before_pass`
    def test_insert_before_pass(self, pipeline):
        """Test the ``insert_before_pass`` function with valid inputs, inserting into the middle of
        the pipeline.
        """
        new_pass_1 = "new_pass_1"
        insert_before_pass(pipeline, ref_pass="pass2", new_pass=new_pass_1)

        pipeline_expected = ["pass1", new_pass_1, "pass2", "pass3"]
        assert pipeline == pipeline_expected

    def test_insert_before_pass_at_beginning(self, pipeline):
        """Test the ``insert_before_pass`` function with valid inputs, inserting at the beginning of
        the pipeline.
        """
        new_pass_1 = "new_pass_1"
        insert_before_pass(pipeline, ref_pass="pass1", new_pass=new_pass_1)

        pipeline_expected = [new_pass_1, "pass1", "pass2", "pass3"]
        assert pipeline == pipeline_expected

    @pytest.mark.parametrize("ref_pass", ["not_a_pass", 1, [], None])
    def test_insert_before_invalid_ref_pass(self, pipeline, ref_pass):
        """Test the ``insert_before_pass`` function with an invalid reference pass name."""
        new_pass = "new_pass_1"

        with pytest.raises(ValueError, match=f"Cannot insert pass '{new_pass}' into pipeline"):
            insert_before_pass(pipeline, ref_pass=ref_pass, new_pass=new_pass)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
