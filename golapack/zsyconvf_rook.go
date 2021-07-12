package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsyconvfrook If parameter WAY = 'C':
// ZSYCONVF_ROOK converts the factorization output format used in
// ZSYTRF_ROOK provided on entry in parameter A into the factorization
// output format used in ZSYTRF_RK (or ZSYTRF_BK) that is stored
// on exit in parameters A and E. IPIV format for ZSYTRF_ROOK and
// ZSYTRF_RK (or ZSYTRF_BK) is the same and is not converted.
//
// If parameter WAY = 'R':
// ZSYCONVF_ROOK performs the conversion in reverse direction, i.e.
// converts the factorization output format used in ZSYTRF_RK
// (or ZSYTRF_BK) provided on entry in parameters A and E into
// the factorization output format used in ZSYTRF_ROOK that is stored
// on exit in parameter A. IPIV format for ZSYTRF_ROOK and
// ZSYTRF_RK (or ZSYTRF_BK) is the same and is not converted.
//
// ZSYCONVF_ROOK can also convert in Hermitian matrix case, i.e. between
// formats used in ZHETRF_ROOK and ZHETRF_RK (or ZHETRF_BK).
func Zsyconvfrook(uplo, way byte, n *int, a *mat.CMatrix, lda *int, e *mat.CVector, ipiv *[]int, info *int) {
	var convert, upper bool
	var zero complex128
	var i, ip, ip2 int

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
		gltest.Xerbla([]byte("ZSYCONVF_ROOK"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if upper {
		//        Begin A is UPPER
		if convert {
			//           Convert A (A is upper)
			//
			//
			//           Convert VALUE
			//
			//           Assign superdiagonal entries of D to array E and zero out
			//           corresponding entries in input storage A
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
			//
			//           Apply permutations to submatrices of upper part of A
			//           in factorization order where i decreases from N to 1
			i = (*n)
			for i >= 1 {
				if (*ipiv)[i-1] > 0 {
					//                 1-by-1 pivot interchange
					//
					//                 Swap rows i and IPIV(i) in A(1:i,N-i:N)
					ip = (*ipiv)[i-1]
					if i < (*n) {
						if ip != i {
							goblas.Zswap((*n)-i, a.CVector(i-1, i, *lda), a.CVector(ip-1, i, *lda))
						}
					}

				} else {
					//                 2-by-2 pivot interchange
					//
					//                 Swap rows i and IPIV(i) and i-1 and IPIV(i-1)
					//                 in A(1:i,N-i:N)
					ip = -(*ipiv)[i-1]
					ip2 = -(*ipiv)[i-1-1]
					if i < (*n) {
						if ip != i {
							goblas.Zswap((*n)-i, a.CVector(i-1, i, *lda), a.CVector(ip-1, i, *lda))
						}
						if ip2 != (i - 1) {
							goblas.Zswap((*n)-i, a.CVector(i-1-1, i, *lda), a.CVector(ip2-1, i, *lda))
						}
					}
					i = i - 1

				}
				i = i - 1
			}

		} else {
			//           Revert A (A is upper)
			//
			//
			//           Revert PERMUTATIONS
			//
			//           Apply permutations to submatrices of upper part of A
			//           in reverse factorization order where i increases from 1 to N
			i = 1
			for i <= (*n) {
				if (*ipiv)[i-1] > 0 {
					//                 1-by-1 pivot interchange
					//
					//                 Swap rows i and IPIV(i) in A(1:i,N-i:N)
					ip = (*ipiv)[i-1]
					if i < (*n) {
						if ip != i {
							goblas.Zswap((*n)-i, a.CVector(ip-1, i, *lda), a.CVector(i-1, i, *lda))
						}
					}

				} else {
					//                 2-by-2 pivot interchange
					//
					//                 Swap rows i-1 and IPIV(i-1) and i and IPIV(i)
					//                 in A(1:i,N-i:N)
					i = i + 1
					ip = -(*ipiv)[i-1]
					ip2 = -(*ipiv)[i-1-1]
					if i < (*n) {
						if ip2 != (i - 1) {
							goblas.Zswap((*n)-i, a.CVector(ip2-1, i, *lda), a.CVector(i-1-1, i, *lda))
						}
						if ip != i {
							goblas.Zswap((*n)-i, a.CVector(ip-1, i, *lda), a.CVector(i-1, i, *lda))
						}
					}

				}
				i = i + 1
			}

			//           Revert VALUE
			//           Assign superdiagonal entries of D from array E to
			//           superdiagonal entries of A.
			i = (*n)
			for i > 1 {
				if (*ipiv)[i-1] < 0 {
					a.Set(i-1-1, i-1, e.Get(i-1))
					i = i - 1
				}
				i = i - 1
			}

			//        End A is UPPER
		}

	} else {
		//        Begin A is LOWER
		if convert {
			//           Convert A (A is lower)
			//
			//
			//           Convert VALUE
			//           Assign subdiagonal entries of D to array E and zero out
			//           corresponding entries in input storage A
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
			//
			//           Apply permutations to submatrices of lower part of A
			//           in factorization order where i increases from 1 to N
			i = 1
			for i <= (*n) {
				if (*ipiv)[i-1] > 0 {
					//                 1-by-1 pivot interchange
					//
					//                 Swap rows i and IPIV(i) in A(i:N,1:i-1)
					ip = (*ipiv)[i-1]
					if i > 1 {
						if ip != i {
							goblas.Zswap(i-1, a.CVector(i-1, 0, *lda), a.CVector(ip-1, 0, *lda))
						}
					}

				} else {
					//                 2-by-2 pivot interchange
					//
					//                 Swap rows i and IPIV(i) and i+1 and IPIV(i+1)
					//                 in A(i:N,1:i-1)
					ip = -(*ipiv)[i-1]
					ip2 = -(*ipiv)[i]
					if i > 1 {
						if ip != i {
							goblas.Zswap(i-1, a.CVector(i-1, 0, *lda), a.CVector(ip-1, 0, *lda))
						}
						if ip2 != (i + 1) {
							goblas.Zswap(i-1, a.CVector(i, 0, *lda), a.CVector(ip2-1, 0, *lda))
						}
					}
					i = i + 1

				}
				i = i + 1
			}

		} else {
			//           Revert A (A is lower)
			//
			//
			//           Revert PERMUTATIONS
			//
			//           Apply permutations to submatrices of lower part of A
			//           in reverse factorization order where i decreases from N to 1
			i = (*n)
			for i >= 1 {
				if (*ipiv)[i-1] > 0 {
					//                 1-by-1 pivot interchange
					//
					//                 Swap rows i and IPIV(i) in A(i:N,1:i-1)
					ip = (*ipiv)[i-1]
					if i > 1 {
						if ip != i {
							goblas.Zswap(i-1, a.CVector(ip-1, 0, *lda), a.CVector(i-1, 0, *lda))
						}
					}

				} else {
					//                 2-by-2 pivot interchange
					//
					//                 Swap rows i+1 and IPIV(i+1) and i and IPIV(i)
					//                 in A(i:N,1:i-1)
					i = i - 1
					ip = -(*ipiv)[i-1]
					ip2 = -(*ipiv)[i]
					if i > 1 {
						if ip2 != (i + 1) {
							goblas.Zswap(i-1, a.CVector(ip2-1, 0, *lda), a.CVector(i, 0, *lda))
						}
						if ip != i {
							goblas.Zswap(i-1, a.CVector(ip-1, 0, *lda), a.CVector(i-1, 0, *lda))
						}
					}

				}
				i = i - 1
			}

			//           Revert VALUE
			//           Assign subdiagonal entries of D from array E to
			//           subgiagonal entries of A.
			i = 1
			for i <= (*n)-1 {
				if (*ipiv)[i-1] < 0 {
					a.Set(i, i-1, e.Get(i-1))
					i = i + 1
				}
				i = i + 1
			}

		}

		//        End A is LOWER
	}
}
