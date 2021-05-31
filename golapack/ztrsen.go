package golapack

import (
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
func Ztrsen(job, compq byte, _select []bool, n *int, t *mat.CMatrix, ldt *int, q *mat.CMatrix, ldq *int, w *mat.CVector, m *int, s, sep *float64, work *mat.CVector, lwork, info *int) {
	var lquery, wantbh, wantq, wants, wantsp bool
	var est, one, rnorm, scale, zero float64
	var ierr, k, kase, ks, lwmin, n1, n2, nn int
	rwork := vf(1)
	isave := make([]int, 3)

	zero = 0.0
	one = 1.0

	//     Decode and test the input parameters.
	wantbh = job == 'B'
	wants = job == 'E' || wantbh
	wantsp = job == 'V' || wantbh
	wantq = compq == 'V'

	//     Set M to the number of selected eigenvalues.
	(*m) = 0
	for k = 1; k <= (*n); k++ {
		if _select[k-1] {
			(*m) = (*m) + 1
		}
	}

	n1 = (*m)
	n2 = (*n) - (*m)
	nn = n1 * n2

	(*info) = 0
	lquery = ((*lwork) == -1)

	if wantsp {
		lwmin = maxint(1, 2*nn)
	} else if job == 'N' {
		lwmin = 1
	} else if job == 'E' {
		lwmin = maxint(1, nn)
	}

	if job != 'N' && !wants && !wantsp {
		(*info) = -1
	} else if compq != 'N' && !wantq {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*ldt) < maxint(1, *n) {
		(*info) = -6
	} else if (*ldq) < 1 || (wantq && (*ldq) < (*n)) {
		(*info) = -8
	} else if (*lwork) < lwmin && !lquery {
		(*info) = -14
	}

	if (*info) == 0 {
		work.SetRe(0, float64(lwmin))
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTRSEN"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*m) == (*n) || (*m) == 0 {
		if wants {
			(*s) = one
		}
		if wantsp {
			(*sep) = Zlange('1', n, n, t, ldt, rwork)
		}
		goto label40
	}

	//     Collect the selected eigenvalues at the top left corner of T.
	ks = 0
	for k = 1; k <= (*n); k++ {
		if _select[k-1] {
			ks = ks + 1

			//           Swap the K-th eigenvalue to position KS.
			if k != ks {
				Ztrexc(compq, n, t, ldt, q, ldq, &k, &ks, &ierr)
			}
		}
	}

	if wants {
		//        Solve the Sylvester equation for R:
		//
		//           T11*R - R*T22 = scale*T12
		Zlacpy('F', &n1, &n2, t.Off(0, n1+1-1), ldt, work.CMatrix(n1, opts), &n1)
		Ztrsyl('N', 'N', toPtr(-1), &n1, &n2, t, ldt, t.Off(n1+1-1, n1+1-1), ldt, work.CMatrix(n1, opts), &n1, &scale, &ierr)

		//        Estimate the reciprocal of the condition number of the cluster
		//        of eigenvalues.
		rnorm = Zlange('F', &n1, &n2, work.CMatrix(n1, opts), &n1, rwork)
		if rnorm == zero {
			(*s) = one
		} else {
			(*s) = scale / (math.Sqrt(scale*scale/rnorm+rnorm) * math.Sqrt(rnorm))
		}
	}

	if wantsp {
		//        Estimate sep(T11,T22).
		est = zero
		kase = 0
	label30:
		;
		Zlacn2(&nn, work.Off(nn+1-1), work, &est, &kase, &isave)
		if kase != 0 {
			if kase == 1 {
				//              Solve T11*R - R*T22 = scale*X.
				Ztrsyl('N', 'N', toPtr(-1), &n1, &n2, t, ldt, t.Off(n1+1-1, n1+1-1), ldt, work.CMatrix(n1, opts), &n1, &scale, &ierr)
			} else {
				//              Solve T11**H*R - R*T22**H = scale*X.
				Ztrsyl('C', 'C', toPtr(-1), &n1, &n2, t, ldt, t.Off(n1+1-1, n1+1-1), ldt, work.CMatrix(n1, opts), &n1, &scale, &ierr)
			}
			goto label30
		}

		(*sep) = scale / est
	}

label40:
	;

	//     Copy reordered eigenvalues to W.
	for k = 1; k <= (*n); k++ {
		w.Set(k-1, t.Get(k-1, k-1))
	}

	work.SetRe(0, float64(lwmin))
}
