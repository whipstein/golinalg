package golapack

import (
	"fmt"

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
func Dtgexc(wantq, wantz bool, n int, a, b, q, z *mat.Matrix, ifst, ilst int, work *mat.Vector, lwork int) (ifstOut, ilstOut, info int, err error) {
	var lquery bool
	var zero float64
	var here, lwmin, nbf, nbl, nbnext int

	zero = 0.0
	ifstOut = ifst
	ilstOut = ilst

	//     Decode and test input arguments.
	lquery = (lwork == -1)
	if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if q.Rows < 1 || wantq && (q.Rows < max(1, n)) {
		err = fmt.Errorf("q.Rows < 1 || wantq && (q.Rows < max(1, n)): wantq=%v, q.Rows=%v, n=%v", wantq, q.Rows, n)
	} else if z.Rows < 1 || wantz && (z.Rows < max(1, n)) {
		err = fmt.Errorf("z.Rows < 1 || wantz && (z.Rows < max(1, n)): wantz=%v, z.Rows=%v, n=%v", wantz, z.Rows, n)
	} else if ifstOut < 1 || ifstOut > n {
		err = fmt.Errorf("ifst < 1 || ifst > n: ifst=%v, n=%v", ifstOut, n)
	} else if ilstOut < 1 || ilstOut > n {
		err = fmt.Errorf("ilst < 1 || ilst > n: ilst=%v, n=%v", ilstOut, n)
	}

	if err == nil {
		if n <= 1 {
			lwmin = 1
		} else {
			lwmin = 4*n + 16
		}
		work.Set(0, float64(lwmin))

		if lwork < lwmin && !lquery {
			err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dtgexc", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n <= 1 {
		return
	}

	//     Determine the first row of the specified block and find out
	//     if it is 1-by-1 or 2-by-2.
	if ifstOut > 1 {
		if a.Get(ifstOut-1, ifstOut-1-1) != zero {
			ifstOut = ifstOut - 1
		}
	}
	nbf = 1
	if ifstOut < n {
		if a.Get(ifstOut, ifstOut-1) != zero {
			nbf = 2
		}
	}

	//     Determine the first row of the final block
	//     and find out if it is 1-by-1 or 2-by-2.
	if ilstOut > 1 {
		if a.Get(ilstOut-1, ilstOut-1-1) != zero {
			ilstOut = ilstOut - 1
		}
	}
	nbl = 1
	if ilstOut < n {
		if a.Get(ilstOut, ilstOut-1) != zero {
			nbl = 2
		}
	}
	if ifstOut == ilstOut {
		return
	}

	if ifstOut < ilstOut {
		//        Update ILST.
		if nbf == 2 && nbl == 1 {
			ilstOut = ilstOut - 1
		}
		if nbf == 1 && nbl == 2 {
			ilstOut = ilstOut + 1
		}

		here = ifstOut

	label10:
		;

		//        Swap with next one below.
		if nbf == 1 || nbf == 2 {
			//           Current block either 1-by-1 or 2-by-2.
			nbnext = 1
			if here+nbf+1 <= n {
				if a.Get(here+nbf, here+nbf-1) != zero {
					nbnext = 2
				}
			}
			if info, err = Dtgex2(wantq, wantz, n, a, b, q, z, here, nbf, nbnext, work, lwork); err != nil || info != 0 {
				ilstOut = here
				return
			}
			here = here + nbnext

			//           Test if 2-by-2 block breaks into two 1-by-1 blocks.
			if nbf == 2 {
				if a.Get(here, here-1) == zero {
					nbf = 3
				}
			}

		} else {
			//           Current block consists of two 1-by-1 blocks, each of which
			//           must be swapped individually.
			nbnext = 1
			if here+3 <= n {
				if a.Get(here+3-1, here+2-1) != zero {
					nbnext = 2
				}
			}
			if info, err = Dtgex2(wantq, wantz, n, a, b, q, z, here+1, 1, nbnext, work, lwork); err != nil || info != 0 {
				ilstOut = here
				return
			}
			if nbnext == 1 {
				//              Swap two 1-by-1 blocks.
				if info, err = Dtgex2(wantq, wantz, n, a, b, q, z, here, 1, 1, work, lwork); err != nil || info != 0 {
					ilstOut = here
					return
				}
				here = here + 1

			} else {
				//              Recompute NBNEXT in case of 2-by-2 split.
				if a.Get(here+2-1, here) == zero {
					nbnext = 1
				}
				if nbnext == 2 {
					//                 2-by-2 block did not split.
					if info, err = Dtgex2(wantq, wantz, n, a, b, q, z, here, 1, nbnext, work, lwork); err != nil || info != 0 {
						ilstOut = here
						return
					}
					here = here + 2
				} else {
					//                 2-by-2 block did split.
					if info, err = Dtgex2(wantq, wantz, n, a, b, q, z, here, 1, 1, work, lwork); err != nil || info != 0 {
						ilstOut = here
						return
					}
					here = here + 1
					if info, err = Dtgex2(wantq, wantz, n, a, b, q, z, here, 1, 1, work, lwork); err != nil || info != 0 {
						ilstOut = here
						return
					}
					here = here + 1
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

		//        Swap with next one below.
		if nbf == 1 || nbf == 2 {
			//           Current block either 1-by-1 or 2-by-2.
			nbnext = 1
			if here >= 3 {
				if a.Get(here-1-1, here-2-1) != zero {
					nbnext = 2
				}
			}
			if info, err = Dtgex2(wantq, wantz, n, a, b, q, z, here-nbnext, nbnext, nbf, work, lwork); err != nil || info != 0 {
				ilstOut = here
				return
			}
			here = here - nbnext

			//           Test if 2-by-2 block breaks into two 1-by-1 blocks.
			if nbf == 2 {
				if a.Get(here, here-1) == zero {
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
			if info, err = Dtgex2(wantq, wantz, n, a, b, q, z, here-nbnext, nbnext, 1, work, lwork); err != nil || info != 0 {
				ilstOut = here
				return
			}
			if nbnext == 1 {
				//              Swap two 1-by-1 blocks.
				if info, err = Dtgex2(wantq, wantz, n, a, b, q, z, here, nbnext, 1, work, lwork); err != nil || info != 0 {
					ilstOut = here
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
					if info, err = Dtgex2(wantq, wantz, n, a, b, q, z, here-1, 2, 1, work, lwork); err != nil || info != 0 {
						ilstOut = here
						return
					}
					here = here - 2
				} else {
					//                 2-by-2 block did split.
					if info, err = Dtgex2(wantq, wantz, n, a, b, q, z, here, 1, 1, work, lwork); err != nil || info != 0 {
						ilstOut = here
						return
					}
					here = here - 1
					if info, err = Dtgex2(wantq, wantz, n, a, b, q, z, here, 1, 1, work, lwork); err != nil || info != 0 {
						ilstOut = here
						return
					}
					here = here - 1
				}
			}
		}
		if here > ilstOut {
			goto label20
		}
	}
	ilstOut = here
	work.Set(0, float64(lwmin))

	return
}
