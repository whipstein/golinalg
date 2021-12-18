package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtgsna estimates reciprocal condition numbers for specified
// eigenvalues and/or eigenvectors of a matrix pair (A, B) in
// generalized real Schur canonical form (or of any matrix pair
// (Q*A*Z**T, Q*B*Z**T) with orthogonal matrices Q and Z, where
// Z**T denotes the transpose of Z.
//
// (A, B) must be in generalized real Schur form (as returned by DGGES),
// i.e. A is block upper triangular with 1-by-1 and 2-by-2 diagonal
// blocks. B is upper triangular.
func Dtgsna(job, howmny byte, _select []bool, n int, a, b, vl, vr *mat.Matrix, s, dif *mat.Vector, mm int, work *mat.Vector, lwork int, iwork *[]int) (m int, err error) {
	var lquery, pair, somcon, wantbh, wantdf, wants bool
	var alphai, alphar, alprqt, beta, c1, c2, cond, eps, four, lnrm, one, rnrm, root1, root2, smlnum, tmpii, tmpir, tmpri, tmprr, two, uhav, uhavi, uhbv, uhbvi, zero float64
	var difdri, i, ierr, ifst, ilst, iz, k, ks, lwmin, n1, n2 int

	dummy := vf(1)
	dummy1 := vf(1)

	difdri = 3
	zero = 0.0
	one = 1.0
	two = 2.0
	four = 4.0

	//     Decode and test the input parameters
	wantbh = job == 'B'
	wants = job == 'E' || wantbh
	wantdf = job == 'V' || wantbh

	somcon = howmny == 'S'

	lquery = (lwork == -1)

	if !wants && !wantdf {
		err = fmt.Errorf("!wants && !wantdf: job='%c'", job)
	} else if howmny != 'A' && !somcon {
		err = fmt.Errorf("howmny != 'A' && !somcon: howmny='%c'", howmny)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if wants && vl.Rows < n {
		err = fmt.Errorf("wants && vl.Rows < n: job='%c', vl.Rows=%v, n=%v", job, vl.Rows, n)
	} else if wants && vr.Rows < n {
		err = fmt.Errorf("wants && vr.Rows < n: job='%c', vr.Rows=%v, n=%v", job, vr.Rows, n)
	} else {
		//        Set M to the number of eigenpairs for which condition numbers
		//        are required, and test MM.
		if somcon {
			m = 0
			pair = false
			for k = 1; k <= n; k++ {
				if pair {
					pair = false
				} else {
					if k < n {
						if a.Get(k, k-1) == zero {
							if _select[k-1] {
								m = m + 1
							}
						} else {
							pair = true
							if _select[k-1] || _select[k] {
								m = m + 2
							}
						}
					} else {
						if _select[n-1] {
							m = m + 1
						}
					}
				}
			}
		} else {
			m = n
		}

		if n == 0 {
			lwmin = 1
		} else if job == 'V' || job == 'B' {
			lwmin = 2*n*(n+2) + 16
		} else {
			lwmin = n
		}
		work.Set(0, float64(lwmin))

		if mm < m {
			err = fmt.Errorf("mm < m: mm=%v, m=%v", mm, m)
		} else if lwork < lwmin && !lquery {
			err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dtgsna", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Get machine constants
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum) / eps
	ks = 0
	pair = false

	for k = 1; k <= n; k++ {
		//        Determine whether A(k,k) begins a 1-by-1 or 2-by-2 block.
		if pair {
			pair = false
			goto label20
		} else {
			if k < n {
				pair = a.Get(k, k-1) != zero
			}
		}

		//        Determine whether condition numbers are required for the k-th
		//        eigenpair.
		if somcon {
			if pair {
				if !_select[k-1] && !_select[k] {
					goto label20
				}
			} else {
				if !_select[k-1] {
					goto label20
				}
			}
		}

		ks = ks + 1

		if wants {
			//           Compute the reciprocal condition number of the k-th
			//           eigenvalue.
			if pair {
				//              Complex eigenvalue pair.
				rnrm = Dlapy2(vr.Off(0, ks-1).Vector().Nrm2(n, 1), vr.Off(0, ks).Vector().Nrm2(n, 1))
				lnrm = Dlapy2(vl.Off(0, ks-1).Vector().Nrm2(n, 1), vl.Off(0, ks).Vector().Nrm2(n, 1))
				err = work.Gemv(NoTrans, n, n, one, a, vr.Off(0, ks-1).Vector(), 1, zero, 1)
				tmprr = vl.Off(0, ks-1).Vector().Dot(n, work, 1, 1)
				tmpri = vl.Off(0, ks).Vector().Dot(n, work, 1, 1)
				err = work.Gemv(NoTrans, n, n, one, a, vr.Off(0, ks).Vector(), 1, zero, 1)
				tmpii = vl.Off(0, ks).Vector().Dot(n, work, 1, 1)
				tmpir = vl.Off(0, ks-1).Vector().Dot(n, work, 1, 1)
				uhav = tmprr + tmpii
				uhavi = tmpir - tmpri
				err = work.Gemv(NoTrans, n, n, one, b, vr.Off(0, ks-1).Vector(), 1, zero, 1)
				tmprr = vl.Off(0, ks-1).Vector().Dot(n, work, 1, 1)
				tmpri = vl.Off(0, ks).Vector().Dot(n, work, 1, 1)
				err = work.Gemv(NoTrans, n, n, one, b, vr.Off(0, ks).Vector(), 1, zero, 1)
				tmpii = vl.Off(0, ks).Vector().Dot(n, work, 1, 1)
				tmpir = vl.Off(0, ks-1).Vector().Dot(n, work, 1, 1)
				uhbv = tmprr + tmpii
				uhbvi = tmpir - tmpri
				uhav = Dlapy2(uhav, uhavi)
				uhbv = Dlapy2(uhbv, uhbvi)
				cond = Dlapy2(uhav, uhbv)
				s.Set(ks-1, cond/(rnrm*lnrm))
				s.Set(ks, s.Get(ks-1))

			} else {
				//              Real eigenvalue.
				rnrm = vr.Off(0, ks-1).Vector().Nrm2(n, 1)
				lnrm = vl.Off(0, ks-1).Vector().Nrm2(n, 1)
				err = work.Gemv(NoTrans, n, n, one, a, vr.Off(0, ks-1).Vector(), 1, zero, 1)
				uhav = vl.Off(0, ks-1).Vector().Dot(n, work, 1, 1)
				err = work.Gemv(NoTrans, n, n, one, b, vr.Off(0, ks-1).Vector(), 1, zero, 1)
				uhbv = vl.Off(0, ks-1).Vector().Dot(n, work, 1, 1)
				cond = Dlapy2(uhav, uhbv)
				if cond == zero {
					s.Set(ks-1, -one)
				} else {
					s.Set(ks-1, cond/(rnrm*lnrm))
				}
			}
		}

		if wantdf {
			if n == 1 {
				dif.Set(ks-1, Dlapy2(a.Get(0, 0), b.Get(0, 0)))
				goto label20
			}

			//           Estimate the reciprocal condition number of the k-th
			//           eigenvectors.
			if pair {
				//              Copy the  2-by 2 pencil beginning at (A(k,k), B(k, k)).
				//              Compute the eigenvalue(s) at position K.
				work.Set(0, a.Get(k-1, k-1))
				work.Set(1, a.Get(k, k-1))
				work.Set(2, a.Get(k-1, k))
				work.Set(3, a.Get(k, k))
				work.Set(4, b.Get(k-1, k-1))
				work.Set(5, b.Get(k, k-1))
				work.Set(6, b.Get(k-1, k))
				work.Set(7, b.Get(k, k))
				beta, *dummy1.GetPtr(0), alphar, *dummy.GetPtr(0), alphai = Dlag2(work.Matrix(2, opts), work.Off(4).Matrix(2, opts), smlnum*eps)
				alprqt = one
				c1 = two * (alphar*alphar + alphai*alphai + beta*beta)
				c2 = four * beta * beta * alphai * alphai
				root1 = c1 + math.Sqrt(c1*c1-4.0*c2)
				root2 = c2 / root1
				root1 = root1 / two
				cond = math.Min(math.Sqrt(root1), math.Sqrt(root2))
			}

			//           Copy the matrix (A, B) to the array WORK and swap the
			//           diagonal block beginning at A(k,k) to the (1,1) position.
			Dlacpy(Full, n, n, a, work.Matrix(n, opts))
			Dlacpy(Full, n, n, b, work.Off(n*n).Matrix(n, opts))
			ifst = k
			ilst = 1

			if ifst, ilst, _, err = Dtgexc(false, false, n, work.Matrix(n, opts), work.Off(n*n).Matrix(n, opts), dummy.Matrix(1, opts), dummy1.Matrix(1, opts), ifst, ilst, work.Off(n*n*2), lwork-2*n*n); err != nil {
				panic(err)
			}

			if ierr > 0 {
				//              Ill-conditioned problem - swap rejected.
				dif.Set(ks-1, zero)
			} else {
				//              Reordering successful, solve generalized Sylvester
				//              equation for R and L,
				//                         A22 * R - L * A11 = A12
				//                         B22 * R - L * B11 = B12,
				//              and compute estimate of Difl((A11,B11), (A22, B22)).
				n1 = 1
				if work.Get(1) != zero {
					n1 = 2
				}
				n2 = n - n1
				if n2 == 0 {
					dif.Set(ks-1, cond)
				} else {
					i = n*n + 1
					iz = 2*n*n + 1
					if _, *dif.GetPtr(ks - 1), ierr, err = Dtgsyl(NoTrans, difdri, n2, n1, work.Off(n*n1+n1).Matrix(n, opts), work.Matrix(n, opts), work.Off(n1).Matrix(n, opts), work.Off(n*n1+n1+i-1).Matrix(n, opts), work.Off(i-1).Matrix(n, opts), work.Off(n1+i-1).Matrix(n, opts), work.Off(iz), lwork-2*n*n, iwork); err != nil {
						panic(err)
					}

					if pair {
						dif.Set(ks-1, math.Min(math.Max(one, alprqt)*dif.Get(ks-1), cond))
					}
				}
			}
			if pair {
				dif.Set(ks, dif.Get(ks-1))
			}
		}
		if pair {
			ks = ks + 1
		}

	label20:
	}
	work.Set(0, float64(lwmin))

	return
}
