void matrixMultiply(ArrayMetadata *metadata) {

        int aDim0 = metadata->aDims[0].getLength();
        int aDim1 = metadata->aDims[1].getLength();
        int bDim0 = metadata->bDims[0].getLength();
        int bDim1 = metadata->bDims[1].getLength();
        int cDim0 = metadata->cDims[0].getLength();
        int cDim1 = metadata->cDims[1].getLength();

        float *a = spaceRootContent.a;
        float *b = spaceRootContent.b;
        float *c = spaceRootContent.c;

        for (int i = 0; i < aDim0; i++) {
                int i_c_0 = i * cDim1;
                int i_a_0 = i * aDim1;
                for (int j = 0; j <bDim1; j++) {
                        for (int k = 0; k < bDim0; k++) {
                                int k_b_0 =  k * bDim1;
                                c[i_c_0 + j] += a[i_a_0 + k] * b[k_b_0 + j];
                        }
                }
        }
}
