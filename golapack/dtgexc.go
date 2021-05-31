package golapack

import (
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtgexc reorders the generalized real Schur decomposition of a real
// matrix pair (A,B) using an orthogonal equivalence transformation
//
//                (A, B) = Q * (A, B) * Z**T,
//
// so that the diagonal block of (A, B) with row index IFST is moved
// to row ILST.
//
// (A, B) must be in generalized real Schur canonical form (as returned
// by DGGES), i.e. A is block upper triangular with 1-by-1 and 2-by-2
// diagonal blocks. B is upper triangular.
//
// Optionally, the matrices Q and Z of generalized Schur vectors are
// updated.
//
//        Q(in) * A(in) * Z(in)**T = Q(out) * A(out) * Z(out)**T
//        Q(in) * B(in) * Z(in)**T = Q(out) * B(out) * Z(out)**T
func Dtgexc(wantq, wantz bool, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, q *mat.Matrix, ldq *int, z *mat.Matrix, ldz, ifst, ilst *int, work *mat.Vector, lwork, info *int) {
	var lquery bool
	var zero float64
	var here, lwmin, nbf, nbl, nbnext int

	zero = 0.0

	//     Decode and test input arguments.
	(*info) = 0
	lquery = ((*lwork) == -1)
	if (*n) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -7
	} else if (*ldq) < 1 || wantq && ((*ldq) < maxint(1, *n)) {
		(*info) = -9
	} else if (*ldz) < 1 || wantz && ((*ldz) < maxint(1, *n)) {
		(*info) = -11
	} else if (*ifst) < 1 || (*ifst) > (*n) {
		(*info) = -12
	} else if (*ilst) < 1 || (*ilst) > (*n) {
		(*info) = -13
	}

	if (*info) == 0 {
		if (*n) <= 1 {
			lwmin = 1
		} else {
			lwmin = 4*(*n) + 16
		}
		work.Set(0, float64(lwmin))

		if (*lwork) < lwmin && !lquery {
			(*info) = -15
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DTGEXC"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) <= 1 {
		return
	}

	//     Determine the first row of the specified block and find out
	//     if it is 1-by-1 or 2-by-2.
	if (*ifst) > 1 {
		if a.Get((*ifst)-1, (*ifst)-1-1) != zero {
			(*ifst) = (*ifst) - 1
		}
	}
	nbf = 1
	if (*ifst) < (*n) {
		if a.Get((*ifst)+1-1, (*ifst)-1) != zero {
			nbf = 2
		}
	}

	//     Determine the first row of the final block
	//     and find out if it is 1-by-1 or 2-by-2.
	if (*ilst) > 1 {
		if a.Get((*ilst)-1, (*ilst)-1-1) != zero {
			(*ilst) = (*ilst) - 1
		}
	}
	nbl = 1
	if (*ilst) < (*n) {
		if a.Get((*ilst)+1-1, (*ilst)-1) != zero {
			nbl = 2
		}
	}
	if (*ifst) == (*ilst) {
		return
	}

	if (*ifst) < (*ilst) {
		//        Update ILST.
		if nbf == 2 && nbl == 1 {
			(*ilst) = (*ilst) - 1
		}
		if nbf == 1 && nbl == 2 {
			(*ilst) = (*ilst) + 1
		}

		here = (*ifst)

	label10:
		;

		//        Swap with next one below.
		if nbf == 1 || nbf == 2 {
			//           Current block either 1-by-1 or 2-by-2.
			nbnext = 1
			if here+nbf+1 <= (*n) {
				if a.Get(here+nbf+1-1, here+nbf-1) != zero {
					nbnext = 2
				}
			}
			Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, &here, &nbf, &nbnext, work, lwork, info)
			if (*info) != 0 {
				(*ilst) = here
				return
			}
			here = here + nbnext

			//           Test if 2-by-2 block breaks into two 1-by-1 blocks.
			if nbf == 2 {
				if a.Get(here+1-1, here-1) == zero {
					nbf = 3
				}
			}

		} else {
			//           Current block consists of two 1-by-1 blocks, each of which
			//           must be swapped individually.
			nbnext = 1
			if here+3 <= (*n) {
				if a.Get(here+3-1, here+2-1) != zero {
					nbnext = 2
				}
			}
			Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, toPtr(here+1), func() *int { y := 1; return &y }(), &nbnext, work, lwork, info)
			if (*info) != 0 {
				(*ilst) = here
				return
			}
			if nbnext == 1 {
				//              Swap two 1-by-1 blocks.
				Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, &here, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), work, lwork, info)
				if (*info) != 0 {
					(*ilst) = here
					return
				}
				here = here + 1

			} else {
				//              Recompute NBNEXT in case of 2-by-2 split.
				if a.Get(here+2-1, here+1-1) == zero {
					nbnext = 1
				}
				if nbnext == 2 {
					//                 2-by-2 block did not split.
					Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, &here, func() *int { y := 1; return &y }(), &nbnext, work, lwork, info)
					if (*info) != 0 {
						(*ilst) = here
						return
					}
					here = here + 2
				} else {
					//                 2-by-2 block did split.
					Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, &here, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), work, lwork, info)
					if (*info) != 0 {
						(*ilst) = here
						return
					}
					here = here + 1
					Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, &here, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), work, lwork, info)
					if (*info) != 0 {
						(*ilst) = here
						return
					}
					here = here + 1
				}

			}
		}
		if here < (*ilst) {
			goto label10
		}
	} else {
		here = (*ifst)

	label20:
		;

		//        Swap with next one below.
		if nbf == 1 || nbf == 2 {
			//           Current block either 1-by-1 or 2-by-2.
			nbnext = 1
			if here >= 3 {
				if a.Get(here-1-1, here-2-1) != zero {
					nbnext = 2
				}
			}
			Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, toPtr(here-nbnext), &nbnext, &nbf, work, lwork, info)
			if (*info) != 0 {
				(*ilst) = here
				return
			}
			here = here - nbnext

			//           Test if 2-by-2 block breaks into two 1-by-1 blocks.
			if nbf == 2 {
				if a.Get(here+1-1, here-1) == zero {
					nbf = 3
				}
			}

		} else {
			//           Current block consists of two 1-by-1 blocks, each of which
			//           must be swapped individually.
			nbnext = 1
			if here >= 3 {
				if a.Get(here-1-1, here-2-1) != zero {
					nbnext = 2
				}
			}
			Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, toPtr(here-nbnext), &nbnext, func() *int { y := 1; return &y }(), work, lwork, info)
			if (*info) != 0 {
				(*ilst) = here
				return
			}
			if nbnext == 1 {
				//              Swap two 1-by-1 blocks.
				Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, &here, &nbnext, func() *int { y := 1; return &y }(), work, lwork, info)
				if (*info) != 0 {
					(*ilst) = here
					return
				}
				here = here - 1
			} else {
				//             Recompute NBNEXT in case of 2-by-2 split.
				if a.Get(here-1, here-1-1) == zero {
					nbnext = 1
				}
				if nbnext == 2 {
					//                 2-by-2 block did not split.
					Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, toPtr(here-1), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), work, lwork, info)
					if (*info) != 0 {
						(*ilst) = here
						return
					}
					here = here - 2
				} else {
					//                 2-by-2 block did split.
					Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, &here, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), work, lwork, info)
					if (*info) != 0 {
						(*ilst) = here
						return
					}
					here = here - 1
					Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, &here, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), work, lwork, info)
					if (*info) != 0 {
						(*ilst) = here
						return
					}
					here = here - 1
				}
			}
		}
		if here > (*ilst) {
			goto label20
		}
	}
	(*ilst) = here
	work.Set(0, float64(lwmin))
}
