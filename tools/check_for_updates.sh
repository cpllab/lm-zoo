#!/usr/bin/env bash
# CircleCI build tool: check whether a particular model has been updated in
# this CircleCI build run. If not, exit with error code to abort the build job.
#
# Run from project root with:
#
#   ./tools/check_for_updates.sh <model_name> <base_revision> <current_revision>
#
# where <model_name> corresponds to a directory in the `models` directory of this project.
model_dir="models/$1"
previous_sha="$2"
current_sha="$3"

echo "Checking for changes from $previous_sha...$current_sha in $model_dir"

# Exit with error code if there have been no changes.
[ -z $(git --no-pager diff $previous_sha $current_sha $model_dir) ] && echo "No changes to model $1; aborting job." && circleci-agent step halt
