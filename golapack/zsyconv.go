package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsyconv converts A given by ZHETRF into L and D or vice-versa.
// Get nondiagonal elements of D (returned in workspace) and
// apply or reverse permutation done in TRF.
func Zsyconv(uplo, way byte, n *int, a *mat.CMatrix, lda *int, ipiv *[]int, e *mat.CVector, info *int) {
	var convert, upper bool
	var temp, zero complex128
	var i, ip, j int

	zero = (0.0 + 0.0*1i)

	(*info) = 0
	upper = uplo == 'U'
	convert = way == 'C'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if !convert && way != 'R' {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*lda) < max(1, *n) {
		(*info) = -5
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZSYCONV"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if upper {
		//        A is UPPER
		if convert {
			//           Convert A (A is upper)
			//
			//           Convert VALUE
			i = (*n)
			e.Set(0, zero)
			for i > 1 {
				if (*ipiv)[i-1] < 0 {
					e.Set(i-1, a.Get(i-1-1, i-1))
					e.Set(i-1-1, zero)
					a.Set(i-1-1, i-1, zero)
					i = i - 1
				} else {
					e.Set(i-1, zero)
				}
				i = i - 1
			}

			//           Convert PERMUTATIONS
			i = (*n)
			for i >= 1 {
				if (*ipiv)[i-1] > 0 {
					ip = (*ipiv)[i-1]
					if i < (*n) {
						for j = i + 1; j <= (*n); j++ {
							temp = a.Get(ip-1, j-1)
							a.Set(ip-1, j-1, a.Get(i-1, j-1))
							a.Set(i-1, j-1, temp)
						}
					}
				} else {
					ip = -(*ipiv)[i-1]
					if i < (*n) {
						for j = i + 1; j <= (*n); j++ {
							temp = a.Get(ip-1, j-1)
							a.Set(ip-1, j-1, a.Get(i-1-1, j-1))
							a.Set(i-1-1, j-1, temp)
						}
					}
					i = i - 1
				}
				i = i - 1
			}

		} else {
			//           Revert A (A is upper)
			//
			//           Revert PERMUTATIONS
			i = 1
			for i <= (*n) {
				if (*ipiv)[i-1] > 0 {
					ip = (*ipiv)[i-1]
					if i < (*n) {
						for j = i + 1; j <= (*n); j++ {
							temp = a.Get(ip-1, j-1)
							a.Set(ip-1, j-1, a.Get(i-1, j-1))
							a.Set(i-1, j-1, temp)
						}
					}
				} else {
					ip = -(*ipiv)[i-1]
					i = i + 1
					if i < (*n) {
						for j = i + 1; j <= (*n); j++ {
							temp = a.Get(ip-1, j-1)
							a.Set(ip-1, j-1, a.Get(i-1-1, j-1))
							a.Set(i-1-1, j-1, temp)
						}
					}
				}
				i = i + 1
			}

			//           Revert VALUE
			i = (*n)
			for i > 1 {
				if (*ipiv)[i-1] < 0 {
					a.Set(i-1-1, i-1, e.Get(i-1))
					i = i - 1
				}
				i = i - 1
			}
		}

	} else {
		//        A is LOWER
		if convert {
			//           Convert A (A is lower)
			//
			//           Convert VALUE
			i = 1
			e.Set((*n)-1, zero)
			for i <= (*n) {
				if i < (*n) && (*ipiv)[i-1] < 0 {
					e.Set(i-1, a.Get(i, i-1))
					e.Set(i, zero)
					a.Set(i, i-1, zero)
					i = i + 1
				} else {
					e.Set(i-1, zero)
				}
				i = i + 1
			}

			//           Convert PERMUTATIONS
			i = 1
			for i <= (*n) {
				if (*ipiv)[i-1] > 0 {
					ip = (*ipiv)[i-1]
					if i > 1 {
						for j = 1; j <= i-1; j++ {
							temp = a.Get(ip-1, j-1)
							a.Set(ip-1, j-1, a.Get(i-1, j-1))
							a.Set(i-1, j-1, temp)
						}
					}
				} else {
					ip = -(*ipiv)[i-1]
					if i > 1 {
						for j = 1; j <= i-1; j++ {
							temp = a.Get(ip-1, j-1)
							a.Set(ip-1, j-1, a.Get(i, j-1))
							a.Set(i, j-1, temp)
						}
					}
					i = i + 1
				}
				i = i + 1
			}

		} else {
			//           Revert A (A is lower)
			//
			//           Revert PERMUTATIONS
			i = (*n)
			for i >= 1 {
				if (*ipiv)[i-1] > 0 {
					ip = (*ipiv)[i-1]
					if i > 1 {
						for j = 1; j <= i-1; j++ {
							temp = a.Get(i-1, j-1)
							a.Set(i-1, j-1, a.Get(ip-1, j-1))
							a.Set(ip-1, j-1, temp)
						}
					}
				} else {
					ip = -(*ipiv)[i-1]
					i = i - 1
					if i > 1 {
						for j = 1; j <= i-1; j++ {
							temp = a.Get(i, j-1)
							a.Set(i, j-1, a.Get(ip-1, j-1))
							a.Set(ip-1, j-1, temp)
						}
					}
				}
				i = i - 1
			}

			//           Revert VALUE
			i = 1
			for i <= (*n)-1 {
				if (*ipiv)[i-1] < 0 {
					a.Set(i, i-1, e.Get(i-1))
					i = i + 1
				}
				i = i + 1
			}
		}
	}
}
