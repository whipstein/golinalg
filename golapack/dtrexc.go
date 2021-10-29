package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
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
func Dtrexc(compq byte, n int, t, q *mat.Matrix, ifst, ilst int, work *mat.Vector) (ifstOut, ilstOut, info int, err error) {
	var wantq bool
	var zero float64
	var here, nbf, nbl, nbnext int

	zero = 0.0
	ifstOut = ifst
	ilstOut = ilst

	//     Decode and test the input arguments.
	wantq = compq == 'V'
	if !wantq && compq != 'N' {
		err = fmt.Errorf("!wantq && compq != 'N': compq='%c'", compq)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if t.Rows < max(1, n) {
		err = fmt.Errorf("t.Rows < max(1, n): t.Rows=%v, n=%v", t.Rows, n)
	} else if q.Rows < 1 || (wantq && q.Rows < max(1, n)) {
		err = fmt.Errorf("q.Rows < 1 || (wantq && q.Rows < max(1, n)): compq='%c', q.Rows=%v, n=%v", compq, q.Rows, n)
	} else if (ifstOut < 1 || ifstOut > n) && (n > 0) {
		err = fmt.Errorf("(ifst < 1 || ifst > n) && (n > 0): ifst=%v, n=%v", ifstOut, n)
	} else if (ilstOut < 1 || ilstOut > n) && (n > 0) {
		err = fmt.Errorf("(ilst < 1 || ilst > n) && (n > 0): ilst=%v, n=%v", ilstOut, n)
	}
	if err != nil {
		gltest.Xerbla2("Dtrexc", err)
		return
	}

	//     Quick return if possible
	if n <= 1 {
		return
	}

	//     Determine the first row of specified block
	//     and find out it is 1 by 1 or 2 by 2.
	if ifstOut > 1 {
		if t.Get(ifstOut-1, ifstOut-1-1) != zero {
			ifstOut = ifstOut - 1
		}
	}
	nbf = 1
	if ifstOut < n {
		if t.Get(ifstOut, ifstOut-1) != zero {
			nbf = 2
		}
	}

	//     Determine the first row of the final block
	//     and find out it is 1 by 1 or 2 by 2.
	if ilstOut > 1 {
		if t.Get(ilstOut-1, ilstOut-1-1) != zero {
			ilstOut = ilstOut - 1
		}
	}
	nbl = 1
	if ilstOut < n {
		if t.Get(ilstOut, ilstOut-1) != zero {
			nbl = 2
		}
	}

	if ifstOut == ilstOut {
		return
	}

	if ifstOut < ilstOut {
		//        Update ILST
		if nbf == 2 && nbl == 1 {
			ilstOut = ilstOut - 1
		}
		if nbf == 1 && nbl == 2 {
			ilstOut = ilstOut + 1
		}

		here = ifstOut

	label10:
		;

		//        Swap block with next one below
		if nbf == 1 || nbf == 2 {
			//           Current block either 1 by 1 or 2 by 2
			nbnext = 1
			if here+nbf+1 <= n {
				if t.Get(here+nbf, here+nbf-1) != zero {
					nbnext = 2
				}
			}
			if info = Dlaexc(wantq, n, t, q, here, nbf, nbnext, work); info != 0 {
				ilstOut = here
				return
			}
			here = here + nbnext

			//           Test if 2 by 2 block breaks into two 1 by 1 blocks
			if nbf == 2 {
				if t.Get(here, here-1) == zero {
					nbf = 3
				}
			}

		} else {
			//           Current block consists of two 1 by 1 blocks each of which
			//           must be swapped individually
			nbnext = 1
			if here+3 <= n {
				if t.Get(here+3-1, here+2-1) != zero {
					nbnext = 2
				}
			}
			if info = Dlaexc(wantq, n, t, q, here+1, 1, nbnext, work); info != 0 {
				ilstOut = here
				return
			}
			if nbnext == 1 {
				//              Swap two 1 by 1 blocks, no problems possible
				info = Dlaexc(wantq, n, t, q, here, 1, nbnext, work)
				here = here + 1
			} else {
				//              Recompute NBNEXT in case 2 by 2 split
				if t.Get(here+2-1, here) == zero {
					nbnext = 1
				}
				if nbnext == 2 {
					//                 2 by 2 Block did not split
					if info = Dlaexc(wantq, n, t, q, here, 1, nbnext, work); info != 0 {
						ilstOut = here
						return
					}
					here = here + 2
				} else {
					//                 2 by 2 Block did split
					info = Dlaexc(wantq, n, t, q, here, 1, 1, work)
					info = Dlaexc(wantq, n, t, q, here+1, 1, 1, work)
					here = here + 2
				}
			}
		}
		if here < ilstOut {
			goto label10
		}

	} else {

		here = ifstOut
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
			if info = Dlaexc(wantq, n, t, q, here-nbnext, nbnext, nbf, work); info != 0 {
				ilstOut = here
				return
			}
			here = here - nbnext

			//           Test if 2 by 2 block breaks into two 1 by 1 blocks
			if nbf == 2 {
				if t.Get(here, here-1) == zero {
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
			if info = Dlaexc(wantq, n, t, q, here-nbnext, nbnext, 1, work); info != 0 {
				ilstOut = here
				return
			}
			if nbnext == 1 {
				//              Swap two 1 by 1 blocks, no problems possible
				info = Dlaexc(wantq, n, t, q, here, nbnext, 1, work)
				here = here - 1
			} else {
				//              Recompute NBNEXT in case 2 by 2 split
				if t.Get(here-1, here-1-1) == zero {
					nbnext = 1
				}
				if nbnext == 2 {
					//                 2 by 2 Block did not split
					if info = Dlaexc(wantq, n, t, q, here-1, 2, 1, work); info != 0 {
						ilstOut = here
						return
					}
					here = here - 2
				} else {
					//                 2 by 2 Block did split
					info = Dlaexc(wantq, n, t, q, here, 1, 1, work)
					info = Dlaexc(wantq, n, t, q, here-1, 1, 1, work)
					here = here - 2
				}
			}
		}
		if here > ilstOut {
			goto label20
		}
	}
	ilstOut = here

	return
}
