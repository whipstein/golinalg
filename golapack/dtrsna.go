package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
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
func Dtrsna(job, howmny byte, _select []bool, n *int, t *mat.Matrix, ldt *int, vl *mat.Matrix, ldvl *int, vr *mat.Matrix, ldvr *int, s, sep *mat.Vector, mm, m *int, work *mat.Matrix, ldwork *int, iwork *[]int, info *int) {
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

	(*info) = 0
	if !wants && !wantsp {
		(*info) = -1
	} else if howmny != 'A' && !somcon {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*ldt) < maxint(1, *n) {
		(*info) = -6
	} else if (*ldvl) < 1 || (wants && (*ldvl) < (*n)) {
		(*info) = -8
	} else if (*ldvr) < 1 || (wants && (*ldvr) < (*n)) {
		(*info) = -10
	} else {
		//        Set M to the number of eigenpairs for which condition numbers
		//        are required, and test MM.
		if somcon {
			(*m) = 0
			pair = false
			for k = 1; k <= (*n); k++ {
				if pair {
					pair = false
				} else {
					if k < (*n) {
						if t.Get(k+1-1, k-1) == zero {
							if _select[k-1] {
								(*m) = (*m) + 1
							}
						} else {
							pair = true
							if _select[k-1] || _select[k+1-1] {
								(*m) = (*m) + 2
							}
						}
					} else {
						if _select[(*n)-1] {
							(*m) = (*m) + 1
						}
					}
				}
			}
		} else {
			(*m) = (*n)
		}

		if (*mm) < (*m) {
			(*info) = -13
		} else if (*ldwork) < 1 || (wantsp && (*ldwork) < (*n)) {
			(*info) = -16
		}
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DTRSNA"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if (*n) == 1 {
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
	Dlabad(&smlnum, &bignum)

	ks = 0
	pair = false
	for k = 1; k <= (*n); k++ {
		//        Determine whether T(k,k) begins a 1-by-1 or 2-by-2 block.
		if pair {
			pair = false
			goto label60
		} else {
			if k < (*n) {
				pair = t.Get(k+1-1, k-1) != zero
			}
		}

		//        Determine whether condition numbers are required for the k-th
		//        eigenpair.
		if somcon {
			if pair {
				if !_select[k-1] && !_select[k+1-1] {
					goto label60
				}
			} else {
				if !_select[k-1] {
					goto label60
				}
			}
		}

		ks = ks + 1

		if wants {
			//           Compute the reciprocal condition number of the k-th
			//           eigenvalue.
			if !pair {
				//              Real eigenvalue.
				prod = goblas.Ddot(*n, vr.Vector(0, ks-1), 1, vl.Vector(0, ks-1), 1)
				rnrm = goblas.Dnrm2(*n, vr.Vector(0, ks-1), 1)
				lnrm = goblas.Dnrm2(*n, vl.Vector(0, ks-1), 1)
				s.Set(ks-1, math.Abs(prod)/(rnrm*lnrm))
			} else {
				//              Complex eigenvalue.
				prod1 = goblas.Ddot(*n, vr.Vector(0, ks-1), 1, vl.Vector(0, ks-1), 1)
				prod1 = prod1 + goblas.Ddot(*n, vr.Vector(0, ks+1-1), 1, vl.Vector(0, ks+1-1), 1)
				prod2 = goblas.Ddot(*n, vl.Vector(0, ks-1), 1, vr.Vector(0, ks+1-1), 1)
				prod2 = prod2 - goblas.Ddot(*n, vl.Vector(0, ks+1-1), 1, vr.Vector(0, ks-1), 1)
				rnrm = Dlapy2(toPtrf64(goblas.Dnrm2(*n, vr.Vector(0, ks-1), 1)), toPtrf64(goblas.Dnrm2(*n, vr.Vector(0, ks+1-1), 1)))
				lnrm = Dlapy2(toPtrf64(goblas.Dnrm2(*n, vl.Vector(0, ks-1), 1)), toPtrf64(goblas.Dnrm2(*n, vl.Vector(0, ks+1-1), 1)))
				cond = Dlapy2(&prod1, &prod2) / (rnrm * lnrm)
				s.Set(ks-1, cond)
				s.Set(ks+1-1, cond)
			}
		}

		if wantsp {
			//           Estimate the reciprocal condition number of the k-th
			//           eigenvector.
			//
			//           Copy the matrix T to the array WORK and swap the diagonal
			//           block beginning at T(k,k) to the (1,1) position.
			Dlacpy('F', n, n, t, ldt, work, ldwork)
			ifst = k
			ilst = 1
			Dtrexc('N', n, work, ldwork, dummy.Matrix(1, opts), func() *int { y := 1; return &y }(), &ifst, &ilst, work.Vector(0, (*n)+1-1), &ierr)

			if ierr == 1 || ierr == 2 {
				//              Could not swap because blocks not well separated
				scale = one
				est = bignum
			} else {
				//              Reordering successful
				if work.Get(1, 0) == zero {
					//                 Form C = T22 - lambda*I in WORK(2:N,2:N).
					for i = 2; i <= (*n); i++ {
						work.Set(i-1, i-1, work.Get(i-1, i-1)-work.Get(0, 0))
					}
					n2 = 1
					nn = (*n) - 1
				} else {
					//                 Triangularize the 2 by 2 block by unitary
					//                 transformation U = [  cs   i*ss ]
					//                                    [ i*ss   cs  ].
					//                 such that the (1,1) position of WORK is complex
					//                 eigenvalue lambda with positive imaginary part. (2,2)
					//                 position of WORK is the complex eigenvalue lambda
					//                 with negative imaginary  part.
					mu = math.Sqrt(math.Abs(work.Get(0, 1))) * math.Sqrt(math.Abs(work.Get(1, 0)))
					delta = Dlapy2(&mu, work.GetPtr(1, 0))
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
					for j = 3; j <= (*n); j++ {
						work.Set(1, j-1, cs*work.Get(1, j-1))
						work.Set(j-1, j-1, work.Get(j-1, j-1)-work.Get(0, 0))
					}
					work.Set(1, 1, zero)

					work.Set(0, (*n)+1-1, two*mu)
					for i = 2; i <= (*n)-1; i++ {
						work.Set(i-1, (*n)+1-1, sn*work.Get(0, i+1-1))
					}
					n2 = 2
					nn = 2 * ((*n) - 1)
				}

				//              Estimate norm(inv(C**T))
				est = zero
				kase = 0
			label50:
				;
				Dlacn2(&nn, work.Vector(0, (*n)+2-1), work.Vector(0, (*n)+4-1), iwork, &est, &kase, &isave)
				if kase != 0 {
					if kase == 1 {
						if n2 == 1 {
							//                       Real eigenvalue: solve C**T*x = scale*c.
							Dlaqtr(true, true, toPtr((*n)-1), work.Off(1, 1), ldwork, dummy, &dumm, &scale, work.Vector(0, (*n)+4-1), work.Vector(0, (*n)+6-1), &ierr)
						} else {
							//                       Complex eigenvalue: solve
							//                       C**T*(p+iq) = scale*(c+id) in real arithmetic.
							Dlaqtr(true, false, toPtr((*n)-1), work.Off(1, 1), ldwork, work.Vector(0, (*n)+1-1), &mu, &scale, work.Vector(0, (*n)+4-1), work.Vector(0, (*n)+6-1), &ierr)
						}
					} else {
						if n2 == 1 {
							//                       Real eigenvalue: solve C*x = scale*c.
							Dlaqtr(false, true, toPtr((*n)-1), work.Off(1, 1), ldwork, dummy, &dumm, &scale, work.Vector(0, (*n)+4-1), work.Vector(0, (*n)+6-1), &ierr)
						} else {
							//                       Complex eigenvalue: solve
							//                       C*(p+iq) = scale*(c+id) in real arithmetic.
							Dlaqtr(false, false, toPtr((*n)-1), work.Off(1, 1), ldwork, work.Vector(0, (*n)+1-1), &mu, &scale, work.Vector(0, (*n)+4-1), work.Vector(0, (*n)+6-1), &ierr)

						}
					}

					goto label50
				}
			}

			sep.Set(ks-1, scale/maxf64(est, smlnum))
			if pair {
				sep.Set(ks+1-1, sep.Get(ks-1))
			}
		}

		if pair {
			ks = ks + 1
		}

	label60:
	}
}
