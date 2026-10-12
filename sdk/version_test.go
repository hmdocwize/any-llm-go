package sdk

import (
	"regexp"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestVersionFormat validates the Version const matches the semver pattern, allowing a pre-release suffix for fork tags.
func TestVersionFormat(t *testing.T) {
	t.Parallel()

	versionPattern := `^v\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$`
	regex := regexp.MustCompile(versionPattern)
	require.True(t, regex.MatchString(Version))
}
