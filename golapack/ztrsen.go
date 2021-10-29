package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztrsen reorders the Schur factorization of a complex matrix
// A = Q*T*Q**H, so that a selected cluster of eigenvalues appears in
// the leading positions on the diagonal of the upper triangular matrix
// T, and the leading columns of Q form an orthonormal basis of the
// corresponding right invariant subspace.
//
// Optionally the routine computes the reciprocal condition numbers of
// the cluster of eigenvalues and/or the invariant subspace.
func Ztrsen(job, compq byte, _select []bool, n int, t, q *mat.CMatrix, w *mat.CVector, work *mat.CVector, lwork int) (m int, s, sep float64, err error) {
	var lquery, wantbh, wantq, wants, wantsp bool
	var est, one, rnorm, scale, zero float64
	var k, kase, ks, lwmin, n1, n2, nn int

	rwork := vf(1)
	isave := make([]int, 3)

	zero = 0.0
	one = 1.0
	s = -1
	sep = -1

	//     Decode and test the input parameters.
	wantbh = job == 'B'
	wants = job == 'E' || wantbh
	wantsp = job == 'V' || wantbh
	wantq = compq == 'V'

	//     Set M to the number of selected eigenvalues.
	m = 0
	for k = 1; k <= n; k++ {
		if _select[k-1] {
			m = m + 1
		}
	}

	n1 = m
	n2 = n - m
	nn = n1 * n2

	lquery = (lwork == -1)

	if wantsp {
		lwmin = max(1, 2*nn)
	} else if job == 'N' {
		lwmin = 1
	} else if job == 'E' {
		lwmin = max(1, nn)
	}

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
	} else if lwork < lwmin && !lquery {
		err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
	}

	if err == nil {
		work.SetRe(0, float64(lwmin))
	}

	if err != nil {
		gltest.Xerbla2("Ztrsen", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if m == n || m == 0 {
		if wants {
			s = one
		}
		if wantsp {
			sep = Zlange('1', n, n, t, rwork)
		}
		goto label40
	}

	//     Collect the selected eigenvalues at the top left corner of T.
	ks = 0
	for k = 1; k <= n; k++ {
		if _select[k-1] {
			ks = ks + 1

			//           Swap the K-th eigenvalue to position KS.
			if k != ks {
				if err = Ztrexc(compq, n, t, q, k, ks); err != nil {
					panic(err)
				}
			}
		}
	}

	if wants {
		//        Solve the Sylvester equation for R:
		//
		//           T11*R - R*T22 = scale*T12
		Zlacpy(Full, n1, n2, t.Off(0, n1), work.CMatrix(n1, opts))
		if scale, _, err = Ztrsyl(NoTrans, NoTrans, -1, n1, n2, t, t.Off(n1, n1), work.CMatrix(n1, opts)); err != nil {
			panic(err)
		}

		//        Estimate the reciprocal of the condition number of the cluster
		//        of eigenvalues.
		rnorm = Zlange('F', n1, n2, work.CMatrix(n1, opts), rwork)
		if rnorm == zero {
			s = one
		} else {
			s = scale / (math.Sqrt(scale*scale/rnorm+rnorm) * math.Sqrt(rnorm))
		}
	}

	if wantsp {
		//        Estimate sep(T11,T22).
		est = zero
		kase = 0
	label30:
		;
		est, kase = Zlacn2(nn, work.Off(nn), work, est, kase, &isave)
		if kase != 0 {
			if kase == 1 {
				//              Solve T11*R - R*T22 = scale*X.
				if scale, _, err = Ztrsyl(NoTrans, NoTrans, -1, n1, n2, t, t.Off(n1, n1), work.CMatrix(n1, opts)); err != nil {
					panic(err)
				}
			} else {
				//              Solve T11**H*R - R*T22**H = scale*X.
				if scale, _, err = Ztrsyl(ConjTrans, ConjTrans, -1, n1, n2, t, t.Off(n1, n1), work.CMatrix(n1, opts)); err != nil {
					panic(err)
				}
			}
			goto label30
		}

		sep = scale / est
	}

label40:
	;

	//     Copy reordered eigenvalues to W.
	for k = 1; k <= n; k++ {
		w.Set(k-1, t.Get(k-1, k-1))
	}

	work.SetRe(0, float64(lwmin))

	return
}
