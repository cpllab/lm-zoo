#!/usr/bin/env bash
# CircleCI build tool: check whether a particular model has been updated in
# this CircleCI build run. If not, exit with error code to abort the build job.
#
# Run from project root with:
#
#   ./tools/check_for_updates.sh <model_name>
#
# where <model_name> corresponds to a directory in the `models` directory of this project.
model_dir="models/$1"

# Get previous commit sha from CIRCLE_COMPARE_URL
previous_sha=$(echo "$CIRCLE_COMPARE_URL" | grep -Po "(?<=compare/)[a-f0-9]+(?=\^)")

current_sha=$CIRCLECI_SHA1

echo "Checking for changes from $previous_sha...$current_sha in $model_dir"

# Exit with error code if there have been no changes.
[ -z $(git --no-pager diff $previous_sha $current_sha $model_dir) ] && echo "No changes to model $1; aborting job." && exit 1
