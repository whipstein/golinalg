package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dtrexc reorders the real Schur factorization of a real matrix
// A = Q*T*Q**T, so that the diagonal block of T with row index IFST is
// moved to row ILST.
//
// The real Schur form T is reordered by an orthogonal similarity
// transformation Z**T*T*Z, and optionally the matrix Q of Schur vectors
// is updated by postmultiplying it with Z.
//
// T must be in Schur canonical form (as returned by DHSEQR), that is,
// block upper triangular with 1-by-1 and 2-by-2 diagonal blocks; each
// 2-by-2 diagonal block has its diagonal elements equal and its
// off-diagonal elements of opposite sign.
func Dtrexc(compq byte, n *int, t *mat.Matrix, ldt *int, q *mat.Matrix, ldq, ifst, ilst *int, work *mat.Vector, info *int) {
	var wantq bool
	var zero float64
	var here, nbf, nbl, nbnext int

	zero = 0.0

	//     Decode and test the input arguments.
	(*info) = 0
	wantq = compq == 'V'
	if !wantq && compq != 'N' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*ldt) < maxint(1, *n) {
		(*info) = -4
	} else if (*ldq) < 1 || (wantq && (*ldq) < maxint(1, *n)) {
		(*info) = -6
	} else if ((*ifst) < 1 || (*ifst) > (*n)) && ((*n) > 0) {
		(*info) = -7
	} else if ((*ilst) < 1 || (*ilst) > (*n)) && ((*n) > 0) {
		(*info) = -8
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DTREXC"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) <= 1 {
		return
	}

	//     Determine the first row of specified block
	//     and find out it is 1 by 1 or 2 by 2.
	if (*ifst) > 1 {
		if t.Get((*ifst)-1, (*ifst)-1-1) != zero {
			(*ifst) = (*ifst) - 1
		}
	}
	nbf = 1
	if (*ifst) < (*n) {
		if t.Get((*ifst)+1-1, (*ifst)-1) != zero {
			nbf = 2
		}
	}

	//     Determine the first row of the final block
	//     and find out it is 1 by 1 or 2 by 2.
	if (*ilst) > 1 {
		if t.Get((*ilst)-1, (*ilst)-1-1) != zero {
			(*ilst) = (*ilst) - 1
		}
	}
	nbl = 1
	if (*ilst) < (*n) {
		if t.Get((*ilst)+1-1, (*ilst)-1) != zero {
			nbl = 2
		}
	}

	if (*ifst) == (*ilst) {
		return
	}

	if (*ifst) < (*ilst) {
		//        Update ILST
		if nbf == 2 && nbl == 1 {
			(*ilst) = (*ilst) - 1
		}
		if nbf == 1 && nbl == 2 {
			(*ilst) = (*ilst) + 1
		}

		here = (*ifst)

	label10:
		;

		//        Swap block with next one below
		if nbf == 1 || nbf == 2 {
			//           Current block either 1 by 1 or 2 by 2
			nbnext = 1
			if here+nbf+1 <= (*n) {
				if t.Get(here+nbf+1-1, here+nbf-1) != zero {
					nbnext = 2
				}
			}
			Dlaexc(wantq, n, t, ldt, q, ldq, &here, &nbf, &nbnext, work, info)
			if (*info) != 0 {
				(*ilst) = here
				return
			}
			here = here + nbnext

			//           Test if 2 by 2 block breaks into two 1 by 1 blocks
			if nbf == 2 {
				if t.Get(here+1-1, here-1) == zero {
					nbf = 3
				}
			}

		} else {
			//           Current block consists of two 1 by 1 blocks each of which
			//           must be swapped individually
			nbnext = 1
			if here+3 <= (*n) {
				if t.Get(here+3-1, here+2-1) != zero {
					nbnext = 2
				}
			}
			Dlaexc(wantq, n, t, ldt, q, ldq, toPtr(here+1), func() *int { y := 1; return &y }(), &nbnext, work, info)
			if (*info) != 0 {
				(*ilst) = here
				return
			}
			if nbnext == 1 {
				//              Swap two 1 by 1 blocks, no problems possible
				Dlaexc(wantq, n, t, ldt, q, ldq, &here, func() *int { y := 1; return &y }(), &nbnext, work, info)
				here = here + 1
			} else {
				//              Recompute NBNEXT in case 2 by 2 split
				if t.Get(here+2-1, here+1-1) == zero {
					nbnext = 1
				}
				if nbnext == 2 {
					//                 2 by 2 Block did not split
					Dlaexc(wantq, n, t, ldt, q, ldq, &here, func() *int { y := 1; return &y }(), &nbnext, work, info)
					if (*info) != 0 {
						(*ilst) = here
						return
					}
					here = here + 2
				} else {
					//                 2 by 2 Block did split
					Dlaexc(wantq, n, t, ldt, q, ldq, &here, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), work, info)
					Dlaexc(wantq, n, t, ldt, q, ldq, toPtr(here+1), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), work, info)
					here = here + 2
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

		//        Swap block with next one above
		if nbf == 1 || nbf == 2 {
			//           Current block either 1 by 1 or 2 by 2
			nbnext = 1
			if here >= 3 {
				if t.Get(here-1-1, here-2-1) != zero {
					nbnext = 2
				}
			}
			Dlaexc(wantq, n, t, ldt, q, ldq, toPtr(here-nbnext), &nbnext, &nbf, work, info)
			if (*info) != 0 {
				(*ilst) = here
				return
			}
			here = here - nbnext

			//           Test if 2 by 2 block breaks into two 1 by 1 blocks
			if nbf == 2 {
				if t.Get(here+1-1, here-1) == zero {
					nbf = 3
				}
			}

		} else {
			//           Current block consists of two 1 by 1 blocks each of which
			//           must be swapped individually
			nbnext = 1
			if here >= 3 {
				if t.Get(here-1-1, here-2-1) != zero {
					nbnext = 2
				}
			}
			Dlaexc(wantq, n, t, ldt, q, ldq, toPtr(here-nbnext), &nbnext, func() *int { y := 1; return &y }(), work, info)
			if (*info) != 0 {
				(*ilst) = here
				return
			}
			if nbnext == 1 {
				//              Swap two 1 by 1 blocks, no problems possible
				Dlaexc(wantq, n, t, ldt, q, ldq, &here, &nbnext, func() *int { y := 1; return &y }(), work, info)
				here = here - 1
			} else {
				//              Recompute NBNEXT in case 2 by 2 split
				if t.Get(here-1, here-1-1) == zero {
					nbnext = 1
				}
				if nbnext == 2 {
					//                 2 by 2 Block did not split
					Dlaexc(wantq, n, t, ldt, q, ldq, toPtr(here-1), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), work, info)
					if (*info) != 0 {
						(*ilst) = here
						return
					}
					here = here - 2
				} else {
					//                 2 by 2 Block did split
					Dlaexc(wantq, n, t, ldt, q, ldq, &here, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), work, info)
					Dlaexc(wantq, n, t, ldt, q, ldq, toPtr(here-1), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), work, info)
					here = here - 2
				}
			}
		}
		if here > (*ilst) {
			goto label20
		}
	}
	(*ilst) = here
}
