package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtrsen reorders the real Schur factorization of a real matrix
// A = Q*T*Q**T, so that a selected cluster of eigenvalues appears in
// the leading diagonal blocks of the upper quasi-triangular matrix T,
// and the leading columns of Q form an orthonormal basis of the
// corresponding right invariant subspace.
//
// Optionally the routine computes the reciprocal condition numbers of
// the cluster of eigenvalues and/or the invariant subspace.
//
// T must be in Schur canonical form (as returned by DHSEQR), that is,
// block upper triangular with 1-by-1 and 2-by-2 diagonal blocks; each
// 2-by-2 diagonal block has its diagonal elements equal and its
// off-diagonal elements of opposite sign.
func Dtrsen(job, compq byte, _select []bool, n int, t *mat.Matrix, q *mat.Matrix, wr, wi *mat.Vector, s, sep float64, work *mat.Vector, lwork int, iwork *[]int, liwork int) (m int, sOut, sepOut float64, info int, err error) {
	var lquery, pair, swap, wantbh, wantq, wants, wantsp bool
	var est, one, rnorm, scale, zero float64
	var ierr, k, kase, kk, ks, liwmin, lwmin, n1, n2, nn int

	isave := make([]int, 3)

	zero = 0.0
	one = 1.0
	sOut = s
	sepOut = sep

	//     Decode and test the input parameters
	wantbh = job == 'B'
	wants = job == 'E' || wantbh
	wantsp = job == 'V' || wantbh
	wantq = compq == 'V'

	lquery = (lwork == -1)
	// if job != 'N' && !wants && !wantsp {
	// 	info = -1
	// } else if compq != 'N' && !wantq {
	// 	info = -2
	// } else if n < 0 {
	// 	info = -4
	// } else if t.Rows < max(1, n) {
	// 	info = -6
	// } else if q.Rows < 1 || (wantq && q.Rows < n) {
	// 	info = -8
	// } else {
	// 	//        Set M to the dimension of the specified invariant subspace,
	// 	//        and test LWORK and LIWORK.
	// 	m = 0
	// 	pair = false
	// 	for k = 1; k <= n; k++ {
	// 		if pair {
	// 			pair = false
	// 		} else {
	// 			if k < n {
	// 				if t.Get(k, k-1) == zero {
	// 					if _select[k-1] {
	// 						m = m + 1
	// 					}
	// 				} else {
	// 					pair = true
	// 					if _select[k-1] || _select[k] {
	// 						m = m + 2
	// 					}
	// 				}
	// 			} else {
	// 				if _select[n-1] {
	// 					m = m + 1
	// 				}
	// 			}
	// 		}
	// 	}

	// 	n1 = m
	// 	n2 = n - m
	// 	nn = n1 * n2

	// 	if wantsp {
	// 		lwmin = max(1, 2*nn)
	// 		liwmin = max(1, nn)
	// 	} else if job == 'N' {
	// 		lwmin = max(1, n)
	// 		liwmin = 1
	// 	} else if job == 'E' {
	// 		lwmin = max(1, nn)
	// 		liwmin = 1
	// 	}

	// 	if lwork < lwmin && !lquery {
	// 		info = -15
	// 	} else if liwork < liwmin && !lquery {
	// 		info = -17
	// 	}
	// }

	// if info == 0 {
	// 	work.Set(0, float64(lwmin))
	// 	(*iwork)[0] = liwmin
	// }

	// if info != 0 {
	// 	gltest.Xerbla("Dtrsen", -info)
	// 	return
	// } else if lquery {
	// 	return
	// }
	if job != 'N' && !wants && !wantsp {
		err = fmt.Errorf("job != 'N' && !wants && !wantsp: job='%c'", job)
	} else if compq != 'N' && !wantq {
		err = fmt.Errorf("compq != 'N' && !wantq: compq='%c'", compq)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if t.Rows < max(1, n) {
		err = fmt.Errorf("t.Rows < max(1, n): t.Rows=%v, n=%v", t.Rows, n)
	} else if q.Rows < 1 || (wantq && q.Rows < n) {
		err = fmt.Errorf("q.Rows < 1 || (wantq && q.Rows < n): compq='%c', q.Rows=%v, n=%v", compq, q.Rows, n)
	} else {
		//        Set M to the dimension of the specified invariant subspace,
		//        and test LWORK and LIWORK.
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

		n1 = m
		n2 = n - m
		nn = n1 * n2

		if wantsp {
			lwmin = max(1, 2*nn)
			liwmin = max(1, nn)
		} else if job == 'N' {
			lwmin = max(1, n)
			liwmin = 1
		} else if job == 'E' {
			lwmin = max(1, nn)
			liwmin = 1
		}

		if lwork < lwmin && !lquery {
			err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
			info = -15
		} else if liwork < liwmin && !lquery {
			err = fmt.Errorf("liwork < liwmin && !lquery: liwork=%v, liwmin=%v, lquery=%v", liwork, liwmin, lquery)
			info = -17
		}
	}

	if err == nil {
		work.Set(0, float64(lwmin))
		(*iwork)[0] = liwmin
	}

	if err != nil {
		gltest.Xerbla2("Dtrsen", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible.
	if m == n || m == 0 {
		if wants {
			sOut = one
		}
		if wantsp {
			sepOut = Dlange('1', n, n, t, work)
		}
		goto label40
	}

	//     Collect the selected blocks at the top-left corner of T.
	ks = 0
	pair = false
	for k = 1; k <= n; k++ {
		if pair {
			pair = false
		} else {
			swap = _select[k-1]
			if k < n {
				if t.Get(k, k-1) != zero {
					pair = true
					swap = swap || _select[k]
				}
			}
			if swap {
				ks = ks + 1

				//              Swap the K-th block to position KS.
				ierr = 0
				kk = k
				if k != ks {
					if kk, ks, ierr, err = Dtrexc(compq, n, t, q, kk, ks, work); err != nil {
						panic(err)
					}
				}
				if ierr == 1 || ierr == 2 {
					//                 Blocks too close to swap: exit.
					info = 1
					if wants {
						sOut = zero
					}
					if wantsp {
						sepOut = zero
					}
					goto label40
				}
				if pair {
					ks = ks + 1
				}
			}
		}
	}

	if wants {
		//        Solve Sylvester equation for R:
		//
		//           T11*R - R*T22 = scale*T12
		Dlacpy(Full, n1, n2, t.Off(0, n1), work.Matrix(n1, opts))
		if scale, ierr, err = Dtrsyl(NoTrans, NoTrans, -1, n1, n2, t, t.Off(n1, n1), work.Matrix(n1, opts)); err != nil {
			panic(err)
		}

		//        Estimate the reciprocal of the condition number of the cluster
		//        of eigenvalues.
		rnorm = Dlange('F', n1, n2, work.Matrix(n1, opts), work)
		if rnorm == zero {
			sOut = one
		} else {
			sOut = scale / (math.Sqrt(scale*scale/rnorm+rnorm) * math.Sqrt(rnorm))
		}
	}

	if wantsp {
		//        Estimate sep(T11,T22).
		est = zero
		kase = 0
	label30:
		;
		est, kase = Dlacn2(nn, work.Off(nn), work, iwork, est, kase, &isave)
		if kase != 0 {
			if kase == 1 {
				//              Solve  T11*R - R*T22 = scale*X.
				if scale, ierr, err = Dtrsyl(NoTrans, NoTrans, -1, n1, n2, t, t.Off(n1, n1), work.Matrix(n1, opts)); err != nil {
					panic(err)
				}
			} else {
				//              Solve T11**T*R - R*T22**T = scale*X.
				if scale, ierr, err = Dtrsyl(Trans, Trans, -1, n1, n2, t, t.Off(n1, n1), work.Matrix(n1, opts)); err != nil {
					panic(err)
				}
			}
			goto label30
		}

		sepOut = scale / est
	}

label40:
	;

	//     Store the output eigenvalues in WR and WI.
	for k = 1; k <= n; k++ {
		wr.Set(k-1, t.Get(k-1, k-1))
		wi.Set(k-1, zero)
	}
	for k = 1; k <= n-1; k++ {
		if t.Get(k, k-1) != zero {
			wi.Set(k-1, math.Sqrt(math.Abs(t.Get(k-1, k)))*math.Sqrt(math.Abs(t.Get(k, k-1))))
			wi.Set(k, -wi.Get(k-1))
		}
	}

	work.Set(0, float64(lwmin))
	(*iwork)[0] = liwmin

	return
}
