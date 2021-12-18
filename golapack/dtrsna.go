package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtrsna estimates reciprocal condition numbers for specified
// eigenvalues and/or right eigenvectors of a real upper
// quasi-triangular matrix T (or of any matrix Q*T*Q**T with Q
// orthogonal).
//
// T must be in Schur canonical form (as returned by DHSEQR), that is,
// block upper triangular with 1-by-1 and 2-by-2 diagonal blocks; each
// 2-by-2 diagonal block has its diagonal elements equal and its
// off-diagonal elements of opposite sign.
func Dtrsna(job, howmny byte, _select []bool, n int, t, vl, vr *mat.Matrix, s, sep *mat.Vector, mm int, work *mat.Matrix, iwork *[]int) (m int, err error) {
	var pair, somcon, wantbh, wants, wantsp bool
	var bignum, cond, cs, delta, dumm, eps, est, lnrm, mu, one, prod, prod1, prod2, rnrm, scale, smlnum, sn, two, zero float64
	var i, ierr, ifst, ilst, j, k, kase, ks, n2, nn int

	dummy := vf(1)
	isave := make([]int, 3)

	zero = 0.0
	one = 1.0
	two = 2.0

	//     Decode and test the input parameters
	wantbh = job == 'B'
	wants = job == 'E' || wantbh
	wantsp = job == 'V' || wantbh

	somcon = howmny == 'S'

	if !wants && !wantsp {
		err = fmt.Errorf("!wants && !wantsp: job='%c'", job)
	} else if howmny != 'A' && !somcon {
		err = fmt.Errorf("howmny != 'A' && !somcon: howmny='%c'", howmny)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if t.Rows < max(1, n) {
		err = fmt.Errorf("t.Rows < max(1, n): t.Rows=%v, n=%v", t.Rows, n)
	} else if vl.Rows < 1 || (wants && vl.Rows < n) {
		err = fmt.Errorf("vl.Rows < 1 || (wants && vl.Rows < n): job='%c', vl.Rows=%v, n=%v", job, vl.Rows, n)
	} else if vr.Rows < 1 || (wants && vr.Rows < n) {
		err = fmt.Errorf("vr.Rows < 1 || (wants && vr.Rows < n): job='%c', vr.Rows=%v, n=%v", job, vr.Rows, n)
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
						if t.Get(k, k-1) == zero {
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

		if mm < m {
			err = fmt.Errorf("mm < m: mm=%v, m=%v", mm, m)
		} else if work.Rows < 1 || (wantsp && work.Rows < n) {
			err = fmt.Errorf("work.Rows < 1 || (wantsp && work.Rows < n): job='%c', work.Rows=%v, n=%v", job, work.Rows, n)
		}
	}
	if err != nil {
		gltest.Xerbla2("Dtrsna", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if n == 1 {
		if somcon {
			if !_select[0] {
				return
			}
		}
		if wants {
			s.Set(0, one)
		}
		if wantsp {
			sep.Set(0, math.Abs(t.Get(0, 0)))
		}
		return
	}

	//     Get machine constants
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum) / eps
	bignum = one / smlnum
	smlnum, bignum = Dlabad(smlnum, bignum)

	ks = 0
	pair = false
	for k = 1; k <= n; k++ {
		//        Determine whether T(k,k) begins a 1-by-1 or 2-by-2 block.
		if pair {
			pair = false
			continue
		} else {
			if k < n {
				pair = t.Get(k, k-1) != zero
			}
		}

		//        Determine whether condition numbers are required for the k-th
		//        eigenpair.
		if somcon {
			if pair {
				if !_select[k-1] && !_select[k] {
					continue
				}
			} else {
				if !_select[k-1] {
					continue
				}
			}
		}

		ks = ks + 1

		if wants {
			//           Compute the reciprocal condition number of the k-th
			//           eigenvalue.
			if !pair {
				//              Real eigenvalue.
				prod = vl.Off(0, ks-1).Vector().Dot(n, vr.Off(0, ks-1).Vector(), 1, 1)
				rnrm = vr.Off(0, ks-1).Vector().Nrm2(n, 1)
				lnrm = vl.Off(0, ks-1).Vector().Nrm2(n, 1)
				s.Set(ks-1, math.Abs(prod)/(rnrm*lnrm))
			} else {
				//              Complex eigenvalue.
				prod1 = vl.Off(0, ks-1).Vector().Dot(n, vr.Off(0, ks-1).Vector(), 1, 1)
				prod1 = prod1 + vl.Off(0, ks).Vector().Dot(n, vr.Off(0, ks).Vector(), 1, 1)
				prod2 = vr.Off(0, ks).Vector().Dot(n, vl.Off(0, ks-1).Vector(), 1, 1)
				prod2 = prod2 - vr.Off(0, ks-1).Vector().Dot(n, vl.Off(0, ks).Vector(), 1, 1)
				rnrm = Dlapy2(vr.Off(0, ks-1).Vector().Nrm2(n, 1), vr.Off(0, ks).Vector().Nrm2(n, 1))
				lnrm = Dlapy2(vl.Off(0, ks-1).Vector().Nrm2(n, 1), vl.Off(0, ks).Vector().Nrm2(n, 1))
				cond = Dlapy2(prod1, prod2) / (rnrm * lnrm)
				s.Set(ks-1, cond)
				s.Set(ks, cond)
			}
		}

		if wantsp {
			//           Estimate the reciprocal condition number of the k-th
			//           eigenvector.
			//
			//           Copy the matrix T to the array WORK and swap the diagonal
			//           block beginning at T(k,k) to the (1,1) position.
			Dlacpy(Full, n, n, t, work)
			ifst = k
			ilst = 1
			if ifst, ilst, ierr, err = Dtrexc('N', n, work, dummy.Matrix(1, opts), ifst, ilst, work.Off(0, n).Vector()); err != nil {
				panic(err)
			}

			if ierr == 1 || ierr == 2 {
				//              Could not swap because blocks not well separated
				scale = one
				est = bignum
			} else {
				//              Reordering successful
				if work.Get(1, 0) == zero {
					//                 Form C = T22 - lambda*I in WORK(2:N,2:N).
					for i = 2; i <= n; i++ {
						work.Set(i-1, i-1, work.Get(i-1, i-1)-work.Get(0, 0))
					}
					n2 = 1
					nn = n - 1
				} else {
					//                 Triangularize the 2 by 2 block by unitary
					//                 transformation U = [  cs   i*ss ]
					//                                    [ i*ss   cs  ].
					//                 such that the (1,1) position of WORK is complex
					//                 eigenvalue lambda with positive imaginary part. (2,2)
					//                 position of WORK is the complex eigenvalue lambda
					//                 with negative imaginary  part.
					mu = math.Sqrt(math.Abs(work.Get(0, 1))) * math.Sqrt(math.Abs(work.Get(1, 0)))
					delta = Dlapy2(mu, work.Get(1, 0))
					cs = mu / delta
					sn = -work.Get(1, 0) / delta

					//                 Form
					//
					//                 C**T = WORK(2:N,2:N) + i*[rwork(1) ..... rwork(n-1) ]
					//                                          [   mu                     ]
					//                                          [         ..               ]
					//                                          [             ..           ]
					//                                          [                  mu      ]
					//                 where C**T is transpose of matrix C,
					//                 and RWORK is stored starting in the N+1-st column of
					//                 WORK.
					for j = 3; j <= n; j++ {
						work.Set(1, j-1, cs*work.Get(1, j-1))
						work.Set(j-1, j-1, work.Get(j-1, j-1)-work.Get(0, 0))
					}
					work.Set(1, 1, zero)

					work.Set(0, n, two*mu)
					for i = 2; i <= n-1; i++ {
						work.Set(i-1, n, sn*work.Get(0, i))
					}
					n2 = 2
					nn = 2 * (n - 1)
				}

				//              Estimate norm(inv(C**T))
				est = zero
				kase = 0
			label50:
				;
				est, kase = Dlacn2(nn, work.Off(0, n+2-1).Vector(), work.Off(0, n+4-1).Vector(), iwork, est, kase, &isave)
				if kase != 0 {
					if kase == 1 {
						if n2 == 1 {
							//                       Real eigenvalue: solve C**T*x = scale*c.
							scale, ierr = Dlaqtr(true, true, n-1, work.Off(1, 1), dummy, dumm, work.Off(0, n+4-1).Vector(), work.Off(0, n+6-1).Vector())
						} else {
							//                       Complex eigenvalue: solve
							//                       C**T*(p+iq) = scale*(c+id) in real arithmetic.
							scale, ierr = Dlaqtr(true, false, n-1, work.Off(1, 1), work.Off(0, n).Vector(), mu, work.Off(0, n+4-1).Vector(), work.Off(0, n+6-1).Vector())
						}
					} else {
						if n2 == 1 {
							//                       Real eigenvalue: solve C*x = scale*c.
							scale, ierr = Dlaqtr(false, true, n-1, work.Off(1, 1), dummy, dumm, work.Off(0, n+4-1).Vector(), work.Off(0, n+6-1).Vector())
						} else {
							//                       Complex eigenvalue: solve
							//                       C*(p+iq) = scale*(c+id) in real arithmetic.
							scale, ierr = Dlaqtr(false, false, n-1, work.Off(1, 1), work.Off(0, n).Vector(), mu, work.Off(0, n+4-1).Vector(), work.Off(0, n+6-1).Vector())

						}
					}

					goto label50
				}
			}

			sep.Set(ks-1, scale/math.Max(est, smlnum))
			if pair {
				sep.Set(ks, sep.Get(ks-1))
			}
		}

		if pair {
			ks = ks + 1
		}

	}

	return
}
