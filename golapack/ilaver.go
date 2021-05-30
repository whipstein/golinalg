package golapack

// Ilaver subroutine returns the LAPACK version.
func Ilaver(versMajor *int, versMinor *int, versPatch *int) {
	*versMajor = 3
	*versMinor = 9
	*versPatch = 0
}
