package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Ztgsen reorders the generalized Schur decomposition of a complex
// matrix pair (A, B) (in terms of an unitary equivalence trans-
// formation Q**H * (A, B) * Z), so that a selected cluster of eigenvalues
// appears in the leading diagonal blocks of the pair (A,B). The leading
// columns of Q and Z form unitary bases of the corresponding left and
// right eigenspaces (deflating subspaces). (A, B) must be in
// generalized Schur canonical form, that is, A and B are both upper
// triangular.
//
// ZTGSEN also computes the generalized eigenvalues
//
//          w(j)= ALPHA(j) / BETA(j)
//
// of the reordered matrix pair (A, B).
//
// Optionally, the routine computes estimates of reciprocal condition
// numbers for eigenvalues and eigenspaces. These are Difu[(A11,B11),
// (A22,B22)] and Difl[(A11,B11), (A22,B22)], i.e. the separation(s)
// between the matrix pairs (A11, B11) and (A22,B22) that correspond to
// the selected cluster and the eigenvalues outside the cluster, resp.,
// and norms of "projections" onto left and right eigenspaces w.r.t.
// the selected cluster in the (1,1)-block.
func Ztgsen(ijob *int, wantq, wantz bool, _select []bool, n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, alpha, beta *mat.CVector, q *mat.CMatrix, ldq *int, z *mat.CMatrix, ldz, m *int, pl, pr *float64, dif *mat.Vector, work *mat.CVector, lwork *int, iwork *[]int, liwork, info *int) {
	var lquery, swap, wantd, wantd1, wantd2, wantp bool
	var temp1, temp2 complex128
	var dscale, dsum, one, rdscal, safmin, zero float64
	var i, idifjb, ierr, ijb, k, kase, ks, liwmin, lwmin, mn2, n1, n2 int
	isave := make([]int, 3)

	idifjb = 3
	zero = 0.0
	one = 1.0

	//     Decode and test the input parameters
	(*info) = 0
	lquery = ((*lwork) == -1 || (*liwork) == -1)

	if (*ijob) < 0 || (*ijob) > 5 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -5
	} else if (*lda) < maxint(1, *n) {
		(*info) = -7
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -9
	} else if (*ldq) < 1 || (wantq && (*ldq) < (*n)) {
		(*info) = -13
	} else if (*ldz) < 1 || (wantz && (*ldz) < (*n)) {
		(*info) = -15
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTGSEN"), -(*info))
		return
	}

	ierr = 0

	wantp = (*ijob) == 1 || (*ijob) >= 4
	wantd1 = (*ijob) == 2 || (*ijob) == 4
	wantd2 = (*ijob) == 3 || (*ijob) == 5
	wantd = wantd1 || wantd2

	//     Set M to the dimension of the specified pair of deflating
	//     subspaces.
	(*m) = 0
	if !lquery || (*ijob) != 0 {
		for k = 1; k <= (*n); k++ {
			alpha.Set(k-1, a.Get(k-1, k-1))
			beta.Set(k-1, b.Get(k-1, k-1))
			if k < (*n) {
				if _select[k-1] {
					(*m) = (*m) + 1
				}
			} else {
				if _select[(*n)-1] {
					(*m) = (*m) + 1
				}
			}
		}
	}

	if (*ijob) == 1 || (*ijob) == 2 || (*ijob) == 4 {
		lwmin = maxint(1, 2*(*m)*((*n)-(*m)))
		liwmin = maxint(1, (*n)+2)
	} else if (*ijob) == 3 || (*ijob) == 5 {
		lwmin = maxint(1, 4*(*m)*((*n)-(*m)))
		liwmin = maxint(1, 2*(*m)*((*n)-(*m)), (*n)+2)
	} else {
		lwmin = 1
		liwmin = 1
	}

	work.SetRe(0, float64(lwmin))
	(*iwork)[0] = liwmin

	if (*lwork) < lwmin && !lquery {
		(*info) = -21
	} else if (*liwork) < liwmin && !lquery {
		(*info) = -23
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTGSEN"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible.
	if (*m) == (*n) || (*m) == 0 {
		if wantp {
			(*pl) = one
			(*pr) = one
		}
		if wantd {
			dscale = zero
			dsum = one
			for i = 1; i <= (*n); i++ {
				Zlassq(n, a.CVector(0, i-1), func() *int { y := 1; return &y }(), &dscale, &dsum)
				Zlassq(n, b.CVector(0, i-1), func() *int { y := 1; return &y }(), &dscale, &dsum)
				//Label20:
			}
			dif.Set(0, dscale*math.Sqrt(dsum))
			dif.Set(1, dif.Get(0))
		}
		goto label70
	}

	//     Get machine constant
	safmin = Dlamch(SafeMinimum)

	//     Collect the selected blocks at the top-left corner of (A, B).
	ks = 0
	for k = 1; k <= (*n); k++ {
		swap = _select[k-1]
		if swap {
			ks = ks + 1

			//           Swap the K-th block to position KS. Compute unitary Q
			//           and Z that will swap adjacent diagonal blocks in (A, B).
			if k != ks {
				Ztgexc(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, &k, &ks, &ierr)
			}

			if ierr > 0 {
				//              Swap is rejected: exit.
				(*info) = 1
				if wantp {
					(*pl) = zero
					(*pr) = zero
				}
				if wantd {
					dif.Set(0, zero)
					dif.Set(1, zero)
				}
				goto label70
			}
		}
	}
	if wantp {
		//        Solve generalized Sylvester equation for R and L:
		//                   A11 * R - L * A22 = A12
		//                   B11 * R - L * B22 = B12
		n1 = (*m)
		n2 = (*n) - (*m)
		i = n1 + 1
		Zlacpy('F', &n1, &n2, a.Off(0, i-1), lda, work.CMatrix(n1, opts), &n1)
		Zlacpy('F', &n1, &n2, b.Off(0, i-1), ldb, work.CMatrixOff(n1*n2+1-1, n1, opts), &n1)
		ijb = 0
		Ztgsyl('N', &ijb, &n1, &n2, a, lda, a.Off(i-1, i-1), lda, work.CMatrix(n1, opts), &n1, b, ldb, b.Off(i-1, i-1), ldb, work.CMatrixOff(n1*n2+1-1, n1, opts), &n1, &dscale, dif.GetPtr(0), work.Off(n1*n2*2+1-1), toPtr((*lwork)-2*n1*n2), iwork, &ierr)

		//        Estimate the reciprocal of norms of "projections" onto
		//        left and right eigenspaces
		rdscal = zero
		dsum = one
		Zlassq(toPtr(n1*n2), work, func() *int { y := 1; return &y }(), &rdscal, &dsum)
		(*pl) = rdscal * math.Sqrt(dsum)
		if (*pl) == zero {
			(*pl) = one
		} else {
			(*pl) = dscale / (math.Sqrt(dscale*dscale/(*pl)+(*pl)) * math.Sqrt(*pl))
		}
		rdscal = zero
		dsum = one
		Zlassq(toPtr(n1*n2), work.Off(n1*n2+1-1), func() *int { y := 1; return &y }(), &rdscal, &dsum)
		(*pr) = rdscal * math.Sqrt(dsum)
		if (*pr) == zero {
			(*pr) = one
		} else {
			(*pr) = dscale / (math.Sqrt(dscale*dscale/(*pr)+(*pr)) * math.Sqrt(*pr))
		}
	}
	if wantd {
		//        Compute estimates Difu and Difl.
		if wantd1 {
			n1 = (*m)
			n2 = (*n) - (*m)
			i = n1 + 1
			ijb = idifjb

			//           Frobenius norm-based Difu estimate.
			Ztgsyl('N', &ijb, &n1, &n2, a, lda, a.Off(i-1, i-1), lda, work.CMatrix(n1, opts), &n1, b, ldb, b.Off(i-1, i-1), ldb, work.CMatrixOff(n1*n2+1-1, n1, opts), &n1, &dscale, dif.GetPtr(0), work.Off(n1*n2*2+1-1), toPtr((*lwork)-2*n1*n2), iwork, &ierr)

			//           Frobenius norm-based Difl estimate.
			Ztgsyl('N', &ijb, &n2, &n1, a.Off(i-1, i-1), lda, a, lda, work.CMatrix(n2, opts), &n2, b.Off(i-1, i-1), ldb, b, ldb, work.CMatrixOff(n1*n2+1-1, n2, opts), &n2, &dscale, dif.GetPtr(1), work.Off(n1*n2*2+1-1), toPtr((*lwork)-2*n1*n2), iwork, &ierr)
		} else {
			//           Compute 1-norm-based estimates of Difu and Difl using
			//           reversed communication with ZLACN2. In each step a
			//           generalized Sylvester equation or a transposed variant
			//           is solved.
			kase = 0
			n1 = (*m)
			n2 = (*n) - (*m)
			i = n1 + 1
			ijb = 0
			mn2 = 2 * n1 * n2

			//           1-norm-based estimate of Difu.
		label40:
			;
			Zlacn2(&mn2, work.Off(mn2+1-1), work, dif.GetPtr(0), &kase, &isave)
			if kase != 0 {
				if kase == 1 {
					//                 Solve generalized Sylvester equation
					Ztgsyl('N', &ijb, &n1, &n2, a, lda, a.Off(i-1, i-1), lda, work.CMatrix(n1, opts), &n1, b, ldb, b.Off(i-1, i-1), ldb, work.CMatrixOff(n1*n2+1-1, n1, opts), &n1, &dscale, dif.GetPtr(0), work.Off(n1*n2*2+1-1), toPtr((*lwork)-2*n1*n2), iwork, &ierr)
				} else {
					//                 Solve the transposed variant.
					Ztgsyl('C', &ijb, &n1, &n2, a, lda, a.Off(i-1, i-1), lda, work.CMatrix(n1, opts), &n1, b, ldb, b.Off(i-1, i-1), ldb, work.CMatrixOff(n1*n2+1-1, n1, opts), &n1, &dscale, dif.GetPtr(0), work.Off(n1*n2*2+1-1), toPtr((*lwork)-2*n1*n2), iwork, &ierr)
				}
				goto label40
			}
			dif.Set(0, dscale/dif.Get(0))
			//
			//           1-norm-based estimate of Difl.
			//
		label50:
			;
			Zlacn2(&mn2, work.Off(mn2+1-1), work, dif.GetPtr(1), &kase, &isave)
			if kase != 0 {
				if kase == 1 {
					//                 Solve generalized Sylvester equation
					Ztgsyl('N', &ijb, &n2, &n1, a.Off(i-1, i-1), lda, a, lda, work.CMatrix(n2, opts), &n2, b.Off(i-1, i-1), ldb, b, ldb, work.CMatrixOff(n1*n2+1-1, n2, opts), &n2, &dscale, dif.GetPtr(1), work.Off(n1*n2*2+1-1), toPtr((*lwork)-2*n1*n2), iwork, &ierr)
				} else {
					//                 Solve the transposed variant.
					Ztgsyl('C', &ijb, &n2, &n1, a.Off(i-1, i-1), lda, a, lda, work.CMatrix(n2, opts), &n2, b, ldb, b.Off(i-1, i-1), ldb, work.CMatrixOff(n1*n2+1-1, n2, opts), &n2, &dscale, dif.GetPtr(1), work.Off(n1*n2*2+1-1), toPtr((*lwork)-2*n1*n2), iwork, &ierr)
				}
				goto label50
			}
			dif.Set(1, dscale/dif.Get(1))
		}
	}

	//     If B(K,K) is complex, make it real and positive (normalization
	//     of the generalized Schur form) and Store the generalized
	//     eigenvalues of reordered pair (A, B)
	for k = 1; k <= (*n); k++ {
		dscale = b.GetMag(k-1, k-1)
		if dscale > safmin {
			temp1 = b.GetConj(k-1, k-1) / complex(dscale, 0)
			temp2 = b.Get(k-1, k-1) / complex(dscale, 0)
			b.SetRe(k-1, k-1, dscale)
			goblas.Zscal(toPtr((*n)-k), &temp1, b.CVector(k-1, k+1-1), ldb)
			goblas.Zscal(toPtr((*n)-k+1), &temp1, a.CVector(k-1, k-1), lda)
			if wantq {
				goblas.Zscal(n, &temp2, q.CVector(0, k-1), func() *int { y := 1; return &y }())
			}
		} else {
			b.SetRe(k-1, k-1, zero)
		}

		alpha.Set(k-1, a.Get(k-1, k-1))
		beta.Set(k-1, b.Get(k-1, k-1))

	}

label70:
	;

	work.SetRe(0, float64(lwmin))
	(*iwork)[0] = liwmin
}
