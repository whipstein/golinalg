package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// DsyconvfRook
// If parameter WAY = 'C':
// DSYCONVF_ROOK converts the factorization output format used in
// DSYTRF_ROOK provided on entry in parameter A into the factorization
// output format used in DSYTRF_RK (or DSYTRF_BK) that is stored
// on exit in parameters A and E. IPIV format for DSYTRF_ROOK and
// DSYTRF_RK (or DSYTRF_BK) is the same and is not converted.
//
// If parameter WAY = 'R':
// DSYCONVF_ROOK performs the conversion in reverse direction, i.e.
// converts the factorization output format used in DSYTRF_RK
// (or DSYTRF_BK) provided on entry in parameters A and E into
// the factorization output format used in DSYTRF_ROOK that is stored
// on exit in parameter A. IPIV format for DSYTRF_ROOK and
// DSYTRF_RK (or DSYTRF_BK) is the same and is not converted.
func DsyconvfRook(uplo, way byte, n *int, a *mat.Matrix, lda *int, e *mat.Vector, ipiv *[]int, info *int) {
	var convert, upper bool
	var zero float64
	var i, ip, ip2 int

	zero = 0.0

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
		gltest.Xerbla([]byte("DSYCONVF_ROOK"), -(*info))
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
							goblas.Dswap((*n)-i, a.Vector(i-1, i, *lda), a.Vector(ip-1, i, *lda))
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
							goblas.Dswap((*n)-i, a.Vector(i-1, i, *lda), a.Vector(ip-1, i, *lda))
						}
						if ip2 != (i - 1) {
							goblas.Dswap((*n)-i, a.Vector(i-1-1, i, *lda), a.Vector(ip2-1, i, *lda))
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
							goblas.Dswap((*n)-i, a.Vector(ip-1, i, *lda), a.Vector(i-1, i, *lda))
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
							goblas.Dswap((*n)-i, a.Vector(ip2-1, i, *lda), a.Vector(i-1-1, i, *lda))
						}
						if ip != i {
							goblas.Dswap((*n)-i, a.Vector(ip-1, i, *lda), a.Vector(i-1, i, *lda))
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
							goblas.Dswap(i-1, a.Vector(i-1, 0, *lda), a.Vector(ip-1, 0, *lda))
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
							goblas.Dswap(i-1, a.Vector(i-1, 0, *lda), a.Vector(ip-1, 0, *lda))
						}
						if ip2 != (i + 1) {
							goblas.Dswap(i-1, a.Vector(i, 0, *lda), a.Vector(ip2-1, 0, *lda))
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
							goblas.Dswap(i-1, a.Vector(ip-1, 0, *lda), a.Vector(i-1, 0, *lda))
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
							goblas.Dswap(i-1, a.Vector(ip2-1, 0, *lda), a.Vector(i, 0, *lda))
						}
						if ip != i {
							goblas.Dswap(i-1, a.Vector(ip-1, 0, *lda), a.Vector(i-1, 0, *lda))
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
