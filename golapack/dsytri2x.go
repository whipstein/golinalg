package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dsytri2x computes the inverse of a real symmetric indefinite matrix
// A using the factorization A = U*D*U**T or A = L*D*L**T computed by
// DSYTRF.
func Dsytri2x(uplo byte, n *int, a *mat.Matrix, lda *int, ipiv *[]int, work *mat.Vector, nb *int, info *int) {
	var upper bool
	var ak, akkp1, akp1, d, one, t, u01IJ, u01Ip1J, u11IJ, u11Ip1J, zero float64
	var count, cut, i, iinfo, invd, ip, j, k, nnb, u11 int

	_work := work.Matrix((*n)+(*nb)+1, opts)

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -4
	}

	//     Quick return if possible
	if (*info) != 0 {
		gltest.Xerbla([]byte("DSYTRI2X"), -(*info))
		return
	}
	if (*n) == 0 {
		return
	}

	//     Convert A
	//     Workspace got Non-diag elements of D
	Dsyconv(uplo, 'C', n, a, lda, ipiv, work, &iinfo)

	//     Check that the diagonal matrix D is nonsingular.
	if upper {
		//        Upper triangular storage: examine D from bottom to top
		for (*info) = (*n); (*info) >= 1; (*info)-- {
			if (*ipiv)[(*info)-1] > 0 && a.Get((*info)-1, (*info)-1) == zero {
				return
			}
		}
	} else {
		//        Lower triangular storage: examine D from top to bottom.
		for (*info) = 1; (*info) <= (*n); (*info)++ {
			if (*ipiv)[(*info)-1] > 0 && a.Get((*info)-1, (*info)-1) == zero {
				return
			}
		}
	}
	(*info) = 0

	//  Splitting Workspace
	//     U01 is a block (N,NB+1)
	//     The first element of U01 is in WORK(1,1)
	//     U11 is a block (NB+1,NB+1)
	//     The first element of U11 is in WORK(N+1,1)
	u11 = (*n)
	//     INVD is a block (N,2)
	//     The first element of INVD is in WORK(1,INVD)
	invd = (*nb) + 2
	if upper {
		//        invA = P * inv(U**T)*inv(D)*inv(U)*P**T.
		Dtrtri(uplo, 'U', n, a, lda, info)

		//       inv(D) and inv(D)*inv(U)
		k = 1
		for k <= (*n) {
			if (*ipiv)[k-1] > 0 {
				//           1 x 1 diagonal NNB
				_work.Set(k-1, invd-1, one/a.Get(k-1, k-1))
				_work.Set(k-1, invd+1-1, 0)
				k = k + 1
			} else {
				//           2 x 2 diagonal NNB
				t = _work.Get(k+1-1, 0)
				ak = a.Get(k-1, k-1) / t
				akp1 = a.Get(k+1-1, k+1-1) / t
				akkp1 = _work.Get(k+1-1, 0) / t
				d = t * (ak*akp1 - one)
				_work.Set(k-1, invd-1, akp1/d)
				_work.Set(k+1-1, invd+1-1, ak/d)
				_work.Set(k-1, invd+1-1, -akkp1/d)
				_work.Set(k+1-1, invd-1, -akkp1/d)
				k = k + 2
			}
		}

		//       inv(U**T) = (inv(U))**T
		//
		//       inv(U**T)*inv(D)*inv(U)
		cut = (*n)
		for cut > 0 {
			nnb = (*nb)
			if cut <= nnb {
				nnb = cut
			} else {
				count = 0
				//             count negative elements,
				for i = cut + 1 - nnb; i <= cut; i++ {
					if (*ipiv)[i-1] < 0 {
						count = count + 1
					}
				}
				//             need a even number for a clear cut
				if count%2 == 1 {
					nnb = nnb + 1
				}
			}
			cut = cut - nnb

			//          U01 Block
			for i = 1; i <= cut; i++ {
				for j = 1; j <= nnb; j++ {
					_work.Set(i-1, j-1, a.Get(i-1, cut+j-1))
				}
			}

			//          U11 Block
			for i = 1; i <= nnb; i++ {
				_work.Set(u11+i-1, i-1, one)
				for j = 1; j <= i-1; j++ {
					_work.Set(u11+i-1, j-1, zero)
				}
				for j = i + 1; j <= nnb; j++ {
					_work.Set(u11+i-1, j-1, a.Get(cut+i-1, cut+j-1))
				}
			}

			//          invD*U01
			i = 1
			for i <= cut {
				if (*ipiv)[i-1] > 0 {
					for j = 1; j <= nnb; j++ {
						_work.Set(i-1, j-1, _work.Get(i-1, invd-1)*_work.Get(i-1, j-1))
					}
					i = i + 1
				} else {
					for j = 1; j <= nnb; j++ {
						u01IJ = _work.Get(i-1, j-1)
						u01Ip1J = _work.Get(i+1-1, j-1)
						_work.Set(i-1, j-1, _work.Get(i-1, invd-1)*u01IJ+_work.Get(i-1, invd+1-1)*u01Ip1J)
						_work.Set(i+1-1, j-1, _work.Get(i+1-1, invd-1)*u01IJ+_work.Get(i+1-1, invd+1-1)*u01Ip1J)
					}
					i = i + 2
				}
			}

			//        invD1*U11
			i = 1
			for i <= nnb {
				if (*ipiv)[cut+i-1] > 0 {
					for j = i; j <= nnb; j++ {
						_work.Set(u11+i-1, j-1, _work.Get(cut+i-1, invd-1)*_work.Get(u11+i-1, j-1))
					}
					i = i + 1
				} else {
					for j = i; j <= nnb; j++ {
						u11IJ = _work.Get(u11+i-1, j-1)
						u11Ip1J = _work.Get(u11+i+1-1, j-1)
						_work.Set(u11+i-1, j-1, _work.Get(cut+i-1, invd-1)*_work.Get(u11+i-1, j-1)+_work.Get(cut+i-1, invd+1-1)*_work.Get(u11+i+1-1, j-1))
						_work.Set(u11+i+1-1, j-1, _work.Get(cut+i+1-1, invd-1)*u11IJ+_work.Get(cut+i+1-1, invd+1-1)*u11Ip1J)
					}
					i = i + 2
				}
			}

			//       U11**T*invD1*U11->U11
			goblas.Dtrmm(mat.Left, mat.Upper, mat.Trans, mat.Unit, &nnb, &nnb, &one, a.Off(cut+1-1, cut+1-1), lda, _work.Off(u11+1-1, 0), toPtr((*n)+(*nb)+1))

			for i = 1; i <= nnb; i++ {
				for j = i; j <= nnb; j++ {
					a.Set(cut+i-1, cut+j-1, _work.Get(u11+i-1, j-1))
				}
			}

			//          U01**T*invD*U01->A(CUT+I,CUT+J)
			goblas.Dgemm(mat.Trans, mat.NoTrans, &nnb, &nnb, &cut, &one, a.Off(0, cut+1-1), lda, _work, toPtr((*n)+(*nb)+1), &zero, _work.Off(u11+1-1, 0), toPtr((*n)+(*nb)+1))

			//        U11 =  U11**T*invD1*U11 + U01**T*invD*U01
			for i = 1; i <= nnb; i++ {
				for j = i; j <= nnb; j++ {
					a.Set(cut+i-1, cut+j-1, a.Get(cut+i-1, cut+j-1)+_work.Get(u11+i-1, j-1))
				}
			}

			//        U01 =  U00**T*invD0*U01
			goblas.Dtrmm(mat.Left, mat.UploByte(uplo), mat.Trans, mat.Unit, &cut, &nnb, &one, a, lda, _work, toPtr((*n)+(*nb)+1))

			//        Update U01
			for i = 1; i <= cut; i++ {
				for j = 1; j <= nnb; j++ {
					a.Set(i-1, cut+j-1, _work.Get(i-1, j-1))
				}
			}

			//      Next Block
		}

		//        Apply PERMUTATIONS P and P**T: P * inv(U**T)*inv(D)*inv(U) *P**T
		i = 1
		for i <= (*n) {
			if (*ipiv)[i-1] > 0 {
				ip = (*ipiv)[i-1]
				if i < ip {
					Dsyswapr(uplo, n, a, lda, &i, &ip)
				}
				if i > ip {
					Dsyswapr(uplo, n, a, lda, &ip, &i)
				}
			} else {
				ip = -(*ipiv)[i-1]
				i = i + 1
				if (i - 1) < ip {
					Dsyswapr(uplo, n, a, lda, toPtr(i-1), &ip)
				}
				if (i - 1) > ip {
					Dsyswapr(uplo, n, a, lda, &ip, toPtr(i-1))
				}
			}
			i = i + 1
		}
	} else {

		//        LOWER...
		//
		//        invA = P * inv(U**T)*inv(D)*inv(U)*P**T.
		Dtrtri(uplo, 'U', n, a, lda, info)

		//       inv(D) and inv(D)*inv(U)
		k = (*n)
		for k >= 1 {
			if (*ipiv)[k-1] > 0 {
				//           1 x 1 diagonal NNB
				_work.Set(k-1, invd-1, one/a.Get(k-1, k-1))
				_work.Set(k-1, invd+1-1, 0)
				k = k - 1
			} else {
				//           2 x 2 diagonal NNB
				t = _work.Get(k-1-1, 0)
				ak = a.Get(k-1-1, k-1-1) / t
				akp1 = a.Get(k-1, k-1) / t
				akkp1 = _work.Get(k-1-1, 0) / t
				d = t * (ak*akp1 - one)
				_work.Set(k-1-1, invd-1, akp1/d)
				_work.Set(k-1, invd-1, ak/d)
				_work.Set(k-1, invd+1-1, -akkp1/d)
				_work.Set(k-1-1, invd+1-1, -akkp1/d)
				k = k - 2
			}
		}

		//       inv(U**T) = (inv(U))**T
		//
		//       inv(U**T)*inv(D)*inv(U)
		cut = 0
		for cut < (*n) {
			nnb = (*nb)
			if cut+nnb > (*n) {
				nnb = (*n) - cut
			} else {
				count = 0
				//             count negative elements,
				for i = cut + 1; i <= cut+nnb; i++ {
					if (*ipiv)[i-1] < 0 {
						count = count + 1
					}
				}
				//             need a even number for a clear cut
				if count%2 == 1 {
					nnb = nnb + 1
				}
			}
			//     L21 Block
			for i = 1; i <= (*n)-cut-nnb; i++ {
				for j = 1; j <= nnb; j++ {
					_work.Set(i-1, j-1, a.Get(cut+nnb+i-1, cut+j-1))
				}
			}
			//     L11 Block
			for i = 1; i <= nnb; i++ {
				_work.Set(u11+i-1, i-1, one)
				for j = i + 1; j <= nnb; j++ {
					_work.Set(u11+i-1, j-1, zero)
				}
				for j = 1; j <= i-1; j++ {
					_work.Set(u11+i-1, j-1, a.Get(cut+i-1, cut+j-1))
				}
			}

			//          invD*L21
			i = (*n) - cut - nnb
			for i >= 1 {
				if (*ipiv)[cut+nnb+i-1] > 0 {
					for j = 1; j <= nnb; j++ {
						_work.Set(i-1, j-1, _work.Get(cut+nnb+i-1, invd-1)*_work.Get(i-1, j-1))
					}
					i = i - 1
				} else {
					for j = 1; j <= nnb; j++ {
						u01IJ = _work.Get(i-1, j-1)
						u01Ip1J = _work.Get(i-1-1, j-1)
						_work.Set(i-1, j-1, _work.Get(cut+nnb+i-1, invd-1)*u01IJ+_work.Get(cut+nnb+i-1, invd+1-1)*u01Ip1J)
						_work.Set(i-1-1, j-1, _work.Get(cut+nnb+i-1-1, invd+1-1)*u01IJ+_work.Get(cut+nnb+i-1-1, invd-1)*u01Ip1J)
					}
					i = i - 2
				}
			}

			//        invD1*L11
			i = nnb
			for i >= 1 {
				if (*ipiv)[cut+i-1] > 0 {
					for j = 1; j <= nnb; j++ {
						_work.Set(u11+i-1, j-1, _work.Get(cut+i-1, invd-1)*_work.Get(u11+i-1, j-1))
					}
					i = i - 1
				} else {
					for j = 1; j <= nnb; j++ {
						u11IJ = _work.Get(u11+i-1, j-1)
						u11Ip1J = _work.Get(u11+i-1-1, j-1)
						_work.Set(u11+i-1, j-1, _work.Get(cut+i-1, invd-1)*_work.Get(u11+i-1, j-1)+_work.Get(cut+i-1, invd+1-1)*u11Ip1J)
						_work.Set(u11+i-1-1, j-1, _work.Get(cut+i-1-1, invd+1-1)*u11IJ+_work.Get(cut+i-1-1, invd-1)*u11Ip1J)
					}
					i = i - 2
				}
			}

			//       L11**T*invD1*L11->L11
			goblas.Dtrmm(mat.Left, mat.UploByte(uplo), mat.Trans, mat.Unit, &nnb, &nnb, &one, a.Off(cut+1-1, cut+1-1), lda, _work.Off(u11+1-1, 0), toPtr((*n)+(*nb)+1))

			for i = 1; i <= nnb; i++ {
				for j = 1; j <= i; j++ {
					a.Set(cut+i-1, cut+j-1, _work.Get(u11+i-1, j-1))
				}
			}

			if (cut + nnb) < (*n) {
				//          L21**T*invD2*L21->A(CUT+I,CUT+J)
				goblas.Dgemm(mat.Trans, mat.NoTrans, &nnb, &nnb, toPtr((*n)-nnb-cut), &one, a.Off(cut+nnb+1-1, cut+1-1), lda, _work, toPtr((*n)+(*nb)+1), &zero, _work.Off(u11+1-1, 0), toPtr((*n)+(*nb)+1))

				//        L11 =  L11**T*invD1*L11 + U01**T*invD*U01
				for i = 1; i <= nnb; i++ {
					for j = 1; j <= i; j++ {
						a.Set(cut+i-1, cut+j-1, a.Get(cut+i-1, cut+j-1)+_work.Get(u11+i-1, j-1))
					}
				}

				//        L01 =  L22**T*invD2*L21
				goblas.Dtrmm(mat.Left, mat.UploByte(uplo), mat.Trans, mat.Unit, toPtr((*n)-nnb-cut), &nnb, &one, a.Off(cut+nnb+1-1, cut+nnb+1-1), lda, _work, toPtr((*n)+(*nb)+1))

				//      Update L21
				for i = 1; i <= (*n)-cut-nnb; i++ {
					for j = 1; j <= nnb; j++ {
						a.Set(cut+nnb+i-1, cut+j-1, _work.Get(i-1, j-1))
					}
				}
			} else {
				//        L11 =  L11**T*invD1*L11
				for i = 1; i <= nnb; i++ {
					for j = 1; j <= i; j++ {
						a.Set(cut+i-1, cut+j-1, _work.Get(u11+i-1, j-1))
					}
				}
			}

			//      Next Block
			cut = cut + nnb
		}

		//        Apply PERMUTATIONS P and P**T: P * inv(U**T)*inv(D)*inv(U) *P**T
		i = (*n)
		for i >= 1 {
			if (*ipiv)[i-1] > 0 {
				ip = (*ipiv)[i-1]
				if i < ip {
					Dsyswapr(uplo, n, a, lda, &i, &ip)
				}
				if i > ip {
					Dsyswapr(uplo, n, a, lda, &ip, &i)
				}
			} else {
				ip = -(*ipiv)[i-1]
				if i < ip {
					Dsyswapr(uplo, n, a, lda, &i, &ip)
				}
				if i > ip {
					Dsyswapr(uplo, n, a, lda, &ip, &i)
				}
				i = i - 1
			}
			i = i - 1
		}
	}
}
