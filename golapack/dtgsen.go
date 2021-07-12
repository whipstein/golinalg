package golapack

import (
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtgsen reorders the generalized real Schur decomposition of a real
// matrix pair (A, B) (in terms of an orthonormal equivalence trans-
// formation Q**T * (A, B) * Z), so that a selected cluster of eigenvalues
// appears in the leading diagonal blocks of the upper quasi-triangular
// matrix A and the upper triangular B. The leading columns of Q and
// Z form orthonormal bases of the corresponding left and right eigen-
// spaces (deflating subspaces). (A, B) must be in generalized real
// Schur canonical form (as returned by DGGES), i.e. A is block upper
// triangular with 1-by-1 and 2-by-2 diagonal blocks. B is upper
// triangular.
//
// DTGSEN also computes the generalized eigenvalues
//
//             w(j) = (ALPHAR(j) + i*ALPHAI(j))/BETA(j)
//
// of the reordered matrix pair (A, B).
//
// Optionally, DTGSEN computes the estimates of reciprocal condition
// numbers for eigenvalues and eigenspaces. These are Difu[(A11,B11),
// (A22,B22)] and Difl[(A11,B11), (A22,B22)], i.e. the separation(s)
// between the matrix pairs (A11, B11) and (A22,B22) that correspond to
// the selected cluster and the eigenvalues outside the cluster, resp.,
// and norms of "projections" onto left and right eigenspaces w.r.t.
// the selected cluster in the (1,1)-block.
func Dtgsen(ijob *int, wantq bool, wantz bool, _select []bool, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, alphar, alphai, beta *mat.Vector, q *mat.Matrix, ldq *int, z *mat.Matrix, ldz, m *int, pl, pr *float64, dif, work *mat.Vector, lwork *int, iwork *[]int, liwork, info *int) {
	var lquery, pair, swap, wantd, wantd1, wantd2, wantp bool
	var dscale, dsum, eps, one, rdscal, smlnum, zero float64
	var i, idifjb, ierr, ijb, k, kase, kk, ks, liwmin, lwmin, mn2, n1, n2 int

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
	} else if (*lda) < max(1, *n) {
		(*info) = -7
	} else if (*ldb) < max(1, *n) {
		(*info) = -9
	} else if (*ldq) < 1 || (wantq && (*ldq) < (*n)) {
		(*info) = -14
	} else if (*ldz) < 1 || (wantz && (*ldz) < (*n)) {
		(*info) = -16
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DTGSEN"), -(*info))
		return
	}

	//     Get machine constants
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum) / eps
	ierr = 0

	wantp = (*ijob) == 1 || (*ijob) >= 4
	wantd1 = (*ijob) == 2 || (*ijob) == 4
	wantd2 = (*ijob) == 3 || (*ijob) == 5
	wantd = wantd1 || wantd2

	//     Set M to the dimension of the specified pair of deflating
	//     subspaces.
	(*m) = 0
	pair = false
	if !lquery || (*ijob) != 0 {
		for k = 1; k <= (*n); k++ {
			if pair {
				pair = false
			} else {
				if k < (*n) {
					if a.Get(k, k-1) == zero {
						if _select[k-1] {
							(*m) = (*m) + 1
						}
					} else {
						pair = true
						if _select[k-1] || _select[k] {
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
	}

	if (*ijob) == 1 || (*ijob) == 2 || (*ijob) == 4 {
		lwmin = max(1, 4*(*n)+16, 2*(*m)*((*n)-(*m)))
		liwmin = max(1, (*n)+6)
	} else if (*ijob) == 3 || (*ijob) == 5 {
		lwmin = max(1, 4*(*n)+16, 4*(*m)*((*n)-(*m)))
		liwmin = max(1, 2*(*m)*((*n)-(*m)), (*n)+6)
	} else {
		lwmin = max(1, 4*(*n)+16)
		liwmin = 1
	}

	work.Set(0, float64(lwmin))
	(*iwork)[0] = liwmin

	if (*lwork) < lwmin && !lquery {
		(*info) = -22
	} else if (*liwork) < liwmin && !lquery {
		(*info) = -24
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DTGSEN"), -(*info))
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
				Dlassq(n, a.Vector(0, i-1), func() *int { y := 1; return &y }(), &dscale, &dsum)
				Dlassq(n, b.Vector(0, i-1), func() *int { y := 1; return &y }(), &dscale, &dsum)
			}
			dif.Set(0, dscale*math.Sqrt(dsum))
			dif.Set(1, dif.Get(0))
		}
		goto label60
	}

	//     Collect the selected blocks at the top-left corner of (A, B).
	ks = 0
	pair = false
	for k = 1; k <= (*n); k++ {
		if pair {
			pair = false
		} else {

			swap = _select[k-1]
			if k < (*n) {
				if a.Get(k, k-1) != zero {
					pair = true
					swap = swap || _select[k]
				}
			}

			if swap {
				ks = ks + 1

				//              Swap the K-th block to position KS.
				//              Perform the reordering of diagonal blocks in (A, B)
				//              by orthogonal transformation matrices and update
				//              Q and Z accordingly (if requested):
				kk = k
				if k != ks {
					Dtgexc(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, &kk, &ks, work, lwork, &ierr)
				}

				if ierr > 0 {
					//                 Swap is rejected: exit.
					(*info) = 1
					if wantp {
						(*pl) = zero
						(*pr) = zero
					}
					if wantd {
						dif.Set(0, zero)
						dif.Set(1, zero)
					}
					goto label60
				}

				if pair {
					ks = ks + 1
				}
			}
		}
	}
	if wantp {
		//        Solve generalized Sylvester equation for R and L
		//        and compute PL and PR.
		n1 = (*m)
		n2 = (*n) - (*m)
		i = n1 + 1
		ijb = 0
		Dlacpy('F', &n1, &n2, a.Off(0, i-1), lda, work.Matrix(n1, opts), &n1)
		Dlacpy('F', &n1, &n2, b.Off(0, i-1), ldb, work.MatrixOff(n1*n2, n1, opts), &n1)
		Dtgsyl('N', &ijb, &n1, &n2, a, lda, a.Off(i-1, i-1), lda, work.Matrix(n1, opts), &n1, b, ldb, b.Off(i-1, i-1), ldb, work.MatrixOff(n1*n2, n1, opts), &n1, &dscale, dif.GetPtr(0), work.Off(n1*n2*2), toPtr((*lwork)-2*n1*n2), iwork, &ierr)

		//        Estimate the reciprocal of norms of "projections" onto left
		//        and right eigenspaces.
		rdscal = zero
		dsum = one
		Dlassq(toPtr(n1*n2), work, func() *int { y := 1; return &y }(), &rdscal, &dsum)
		(*pl) = rdscal * math.Sqrt(dsum)
		if (*pl) == zero {
			(*pl) = one
		} else {
			(*pl) = dscale / (math.Sqrt(dscale*dscale/(*pl)+(*pl)) * math.Sqrt(*pl))
		}
		rdscal = zero
		dsum = one
		Dlassq(toPtr(n1*n2), work.Off(n1*n2), func() *int { y := 1; return &y }(), &rdscal, &dsum)
		(*pr) = rdscal * math.Sqrt(dsum)
		if (*pr) == zero {
			(*pr) = one
		} else {
			(*pr) = dscale / (math.Sqrt(dscale*dscale/(*pr)+(*pr)) * math.Sqrt(*pr))
		}
	}

	if wantd {
		//        Compute estimates of Difu and Difl.
		if wantd1 {
			n1 = (*m)
			n2 = (*n) - (*m)
			i = n1 + 1
			ijb = idifjb

			//           Frobenius norm-based Difu-estimate.
			Dtgsyl('N', &ijb, &n1, &n2, a, lda, a.Off(i-1, i-1), lda, work.Matrix(n1, opts), &n1, b, ldb, b.Off(i-1, i-1), ldb, work.MatrixOff(n1*n2, n1, opts), &n1, &dscale, dif.GetPtr(0), work.Off(2*n1*n2), toPtr((*lwork)-2*n1*n2), iwork, &ierr)

			//           Frobenius norm-based Difl-estimate.
			Dtgsyl('N', &ijb, &n2, &n1, a.Off(i-1, i-1), lda, a, lda, work.Matrix(n2, opts), &n2, b.Off(i-1, i-1), ldb, b, ldb, work.MatrixOff(n1*n2, n2, opts), &n2, &dscale, dif.GetPtr(1), work.Off(2*n1*n2), toPtr((*lwork)-2*n1*n2), iwork, &ierr)
		} else {
			//           Compute 1-norm-based estimates of Difu and Difl using
			//           reversed communication with DLACN2. In each step a
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
			Dlacn2(&mn2, work.Off(mn2), work, iwork, dif.GetPtr(0), &kase, &isave)
			if kase != 0 {
				if kase == 1 {
					//                 Solve generalized Sylvester equation.
					Dtgsyl('N', &ijb, &n1, &n2, a, lda, a.Off(i-1, i-1), lda, work.Matrix(n1, opts), &n1, b, ldb, b.Off(i-1, i-1), ldb, work.MatrixOff(n1*n2, n1, opts), &n1, &dscale, dif.GetPtr(0), work.Off(2*n1*n2), toPtr((*lwork)-2*n1*n2), iwork, &ierr)
				} else {
					//                 Solve the transposed variant.
					Dtgsyl('T', &ijb, &n1, &n2, a, lda, a.Off(i-1, i-1), lda, work.Matrix(n1, opts), &n1, b, ldb, b.Off(i-1, i-1), ldb, work.MatrixOff(n1*n2, n1, opts), &n1, &dscale, dif.GetPtr(0), work.Off(2*n1*n2), toPtr((*lwork)-2*n1*n2), iwork, &ierr)
				}
				goto label40
			}
			dif.Set(0, dscale/dif.Get(0))

			//           1-norm-based estimate of Difl.
		label50:
			;
			Dlacn2(&mn2, work.Off(mn2), work, iwork, dif.GetPtr(1), &kase, &isave)
			if kase != 0 {
				if kase == 1 {
					//                 Solve generalized Sylvester equation.
					Dtgsyl('N', &ijb, &n2, &n1, a.Off(i-1, i-1), lda, a, lda, work.Matrix(n2, opts), &n2, b.Off(i-1, i-1), ldb, b, ldb, work.MatrixOff(n1*n2, n2, opts), &n2, &dscale, dif.GetPtr(1), work.Off(2*n1*n2), toPtr((*lwork)-2*n1*n2), iwork, &ierr)
				} else {
					//                 Solve the transposed variant.
					Dtgsyl('T', &ijb, &n2, &n1, a.Off(i-1, i-1), lda, a, lda, work.Matrix(n2, opts), &n2, b.Off(i-1, i-1), ldb, b, ldb, work.MatrixOff(n1*n2, n2, opts), &n2, &dscale, dif.GetPtr(1), work.Off(2*n1*n2), toPtr((*lwork)-2*n1*n2), iwork, &ierr)
				}
				goto label50
			}
			dif.Set(1, dscale/dif.Get(1))

		}
	}

label60:
	;

	//     Compute generalized eigenvalues of reordered pair (A, B) and
	//     normalize the generalized Schur form.
	pair = false
	for k = 1; k <= (*n); k++ {
		if pair {
			pair = false
		} else {

			if k < (*n) {
				if a.Get(k, k-1) != zero {
					pair = true
				}
			}

			if pair {
				//             Compute the eigenvalue(s) at position K.
				work.Set(0, a.Get(k-1, k-1))
				work.Set(1, a.Get(k, k-1))
				work.Set(2, a.Get(k-1, k))
				work.Set(3, a.Get(k, k))
				work.Set(4, b.Get(k-1, k-1))
				work.Set(5, b.Get(k, k-1))
				work.Set(6, b.Get(k-1, k))
				work.Set(7, b.Get(k, k))
				Dlag2(work.Matrix(2, opts), func() *int { y := 2; return &y }(), work.MatrixOff(4, 2, opts), func() *int { y := 2; return &y }(), toPtrf64(smlnum*eps), beta.GetPtr(k-1), beta.GetPtr(k), alphar.GetPtr(k-1), alphar.GetPtr(k), alphai.GetPtr(k-1))
				alphai.Set(k, -alphai.Get(k-1))

			} else {

				if math.Copysign(one, b.Get(k-1, k-1)) < zero {
					//                 If B(K,K) is negative, make it positive
					for i = 1; i <= (*n); i++ {
						a.Set(k-1, i-1, -a.Get(k-1, i-1))
						b.Set(k-1, i-1, -b.Get(k-1, i-1))
						if wantq {
							q.Set(i-1, k-1, -q.Get(i-1, k-1))
						}
					}
				}

				alphar.Set(k-1, a.Get(k-1, k-1))
				alphai.Set(k-1, zero)
				beta.Set(k-1, b.Get(k-1, k-1))

			}
		}
	}

	work.Set(0, float64(lwmin))
	(*iwork)[0] = liwmin
}
